"""
extract.py — stage 3: feature extraction. The MOST important stage.

Two independent branches:
  1) Color (HSV): convert the color frame to HSV and, for each "target color"
     from the profile (config.PROFILES), cut out pixels in the given range -> a binary mask.
  2) Edges (Canny): find boundaries in the gray frame. This is the fallback path for
     tracing paper, where there is no color.

On output — a dict of masks. White (255) = "there is a feature here", black (0) = background.
Each mask is saved to debug, so you can see by eye what exactly was caught.
"""

import cv2
import numpy as np

from src import config, crop


def extract(prepared, profile_name, saver, use_sam=False):
    """
    prepared — the dict from preprocess: {"color": BGR, "gray": gray}.
    profile_name — which set of colors to take from config.PROFILES.
    saver — DebugSaver for intermediate frames.
    use_sam — if True and SAM is available, augment the color masks with SAM segments.

    Returns:
      {
        "color_masks": {"red": {"mask": ndarray, "type": "fault"}, ...},
        "canny": ndarray | None,
        "combined": ndarray,   # union of all color masks (for illustration)
      }
    """
    color_image = prepared["color"]
    gray_image = prepared["gray"]
    profile = config.PROFILES.get(profile_name, {})

    # Convert to HSV once (all colors are cut from it afterwards).
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    color_masks = {}
    # An empty "zero" mask of the right size, into which we accumulate the union.
    combined = np.zeros(gray_image.shape, dtype=np.uint8)

    # --- Branch 1: color ---
    for color_name, spec in profile.items():
        mask = _mask_for_color(hsv, spec["ranges"])
        color_masks[color_name] = {"mask": mask, "type": spec["type"]}
        combined = cv2.bitwise_or(combined, mask)
        saver.save(f"mask_{color_name}", mask)

    # --- Branch 1c (opt.): SAM as an alternative layer-boundary segmenter ---
    # The heavy path, only under the flag. If SAM/torch/checkpoint are unavailable —
    # sam_extract prints a hint and returns None (silent fallback to the classic path).
    if use_sam:
        from src import sam_extract
        sam_masks = sam_extract.extract_color_masks(color_image, profile, saver)
        if sam_masks:
            for color_name, sam_mask in sam_masks.items():
                if color_name in color_masks:
                    # Merge with the color mask of the same class (SAM augments HSV).
                    color_masks[color_name]["mask"] = cv2.bitwise_or(
                        color_masks[color_name]["mask"], sam_mask)
                else:
                    ctype = profile.get(color_name, {}).get("type", "boundary")
                    color_masks[color_name] = {"mask": sam_mask, "type": ctype}
                combined = cv2.bitwise_or(combined, sam_mask)
            saver.save("mask_sam_combined", combined)

    # --- Branch 1b: dark lines (ink/pencil faults) ---
    # On many maps the faults are dark, not colored; HSV does not catch them.
    if config.EXTRACT_DARK_LINES:
        dark = _extract_dark_lines(gray_image)
        color_masks["dark"] = {"mask": dark, "type": "fault_uncertain"}
        combined = cv2.bitwise_or(combined, dark)
        saver.save("mask_dark", dark)

    # --- Branch 2: edges (Canny) ---
    canny = None
    if config.USE_CANNY:
        canny = cv2.Canny(gray_image, config.CANNY_THRESHOLD_LOW, config.CANNY_THRESHOLD_HIGH)
        saver.save("canny", canny)

    # --- Mask out everything outside the map frame (margins, legend, stamp) ---
    if config.MASK_OUTSIDE_FRAME:
        frame_mask = _frame_interior_mask(color_image)
        if frame_mask is not None:
            for spec in color_masks.values():
                spec["mask"] = cv2.bitwise_and(spec["mask"], frame_mask)
            combined = cv2.bitwise_and(combined, frame_mask)
            if canny is not None:
                canny = cv2.bitwise_and(canny, frame_mask)
            saver.save("frame_mask", frame_mask)

    saver.save("mask_combined", combined)

    return {"color_masks": color_masks, "canny": canny, "combined": combined}


def _frame_interior_mask(color_image):
    """
    A binary mask of the map frame's interior (white = inside the frame).
    None if the frame is not found — then nothing is masked (safe fallback).
    """
    quad = crop.find_map_corners(color_image)
    if quad is None:
        return None
    h, w = color_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    return mask


def _extract_dark_lines(gray):
    """
    Extract thin dark lines (probable faults) with the black-hat operation.

    Black-hat = closing(gray) - gray: it highlights dark structures thinner than the
    kernel on a lighter background. Then a threshold -> a binary mask of dark strokes.
    The shape (long/thin vs blobs) is filtered by the cleanup stage.
    """
    k = config.DARK_BLACKHAT_KERNEL
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, config.DARK_THRESHOLD, 255, cv2.THRESH_BINARY)
    return mask


def color_mask(hsv, ranges):
    """Public wrapper over _mask_for_color (needed by legend.py for the same thresholds)."""
    return _mask_for_color(hsv, ranges)


def _mask_for_color(hsv, ranges):
    """
    Build one binary mask for a color that may have SEVERAL HSV ranges
    (for example, red — two ranges at the ends of the hue wheel).
    The range masks are merged with a logical OR.
    """
    h, w = hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for lower, upper in ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        part = cv2.inRange(hsv, lower_np, upper_np)
        mask = cv2.bitwise_or(mask, part)
    return mask
