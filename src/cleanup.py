"""
cleanup.py — stage 4: cleaning up the binary masks.

After HSV the mask is "dirty": single specks from the paper, fragments of letters,
broken lines. Here we tidy it up with three techniques:

  1) OPEN  (erosion+dilation) — removes single white noise dots.
  2) CLOSE (dilation+erosion) — fills small holes and joins line gaps.
  3) Area filter — discards "blobs" smaller than N pixels (letters, specks),
     keeping large elongated objects (fault lines).

On output — the same masks, but clean. Each is saved to debug.
"""

import cv2
import numpy as np

from src import config


def cleanup(extracted, saver):
    """
    extracted — the dict from extract: {"color_masks", "canny", "combined"}.
    Returns a structure with cleaned masks:
      {"color_masks": {name: {"mask": clean, "type": ...}}, "combined": clean, "canny": clean|None}
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE),
    )

    clean_color_masks = {}
    combined = None

    # Clean each color mask separately.
    for color_name, spec in extracted["color_masks"].items():
        # For faults (lines) we enable the "bridge" to stitch broken strokes.
        is_fault = spec["type"].startswith("fault")
        # Dark lines (fault_uncertain) are noisy: additionally keep only the
        # long thin components, discarding letters/hatching/fills.
        line_only = spec["type"] == "fault_uncertain"
        clean = _clean_mask(spec["mask"], kernel, bridge=is_fault, line_only=line_only)
        clean_color_masks[color_name] = {"mask": clean, "type": spec["type"]}
        saver.save(f"clean_{color_name}", clean)

        # Rebuild the shared clean mask from the cleaned pieces.
        if combined is None:
            combined = clean.copy()
        else:
            combined = cv2.bitwise_or(combined, clean)

    # If there were no color masks (the pencil profile) — take Canny edges as the base.
    clean_canny = None
    if extracted["canny"] is not None:
        # For edges we don't discard small stuff so aggressively — only close the gaps.
        clean_canny = cv2.morphologyEx(extracted["canny"], cv2.MORPH_CLOSE, kernel,
                                       iterations=config.MORPH_CLOSE_ITERATIONS)
        saver.save("clean_canny", clean_canny)

    if combined is None:
        # No color at all — the shared mask becomes the cleaned Canny (or empty).
        h, w = extracted["combined"].shape[:2]
        combined = clean_canny if clean_canny is not None else np.zeros((h, w), dtype=np.uint8)

    saver.save("clean_combined", combined)

    return {"color_masks": clean_color_masks, "combined": combined, "canny": clean_canny}


def _clean_mask(mask, kernel, bridge=False, line_only=False):
    """
    Apply OPEN -> CLOSE -> (opt. BRIDGE) -> small-component filter to one mask.
    bridge=True adds one more CLOSE with a larger kernel, to stitch broken lines.
    line_only=True additionally keeps only long thin components (for dark lines:
    cuts off letters, relief hatching and fills).
    """
    # Edge-stain guard — EARLY, before morphology: while the aging stain is still one
    # connected blob touching the edge, it is removed whole. If we wait until the area
    # filter, OPEN will fragment the stain into pieces, and some will "detach" from the
    # edge and survive.
    if config.DROP_BORDER_TOUCHING:
        mask = _drop_border_components(mask)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                              iterations=config.MORPH_OPEN_ITERATIONS)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel,
                              iterations=config.MORPH_CLOSE_ITERATIONS)

    if bridge and config.BRIDGE_KERNEL_SIZE:
        bridge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.BRIDGE_KERNEL_SIZE, config.BRIDGE_KERNEL_SIZE),
        )
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, bridge_kernel,
                                  iterations=config.BRIDGE_ITERATIONS)

    if line_only:
        return _keep_line_like(closed)
    return _remove_small_components(closed, config.MIN_COMPONENT_AREA)


def _border_tolerance(mask):
    """How many pixels from the edge count as a "touch" (fraction of the long side)."""
    h, w = mask.shape[:2]
    return max(1, int(round(config.BORDER_TOUCH_TOLERANCE_FRAC * max(h, w))))


def _apply_label_keep(labels, keep):
    """Build a mask from the labels to keep (vectorized, without a per-pixel loop)."""
    keep[0] = False  # label 0 is the background, never keep it
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _border_touch_flags(stats, w_img, h_img, tol):
    """A boolean vector over labels: True where the component bbox touches the image edge."""
    x = stats[:, cv2.CC_STAT_LEFT]
    y = stats[:, cv2.CC_STAT_TOP]
    w = stats[:, cv2.CC_STAT_WIDTH]
    h = stats[:, cv2.CC_STAT_HEIGHT]
    return ((x <= tol) | (y <= tol)
            | ((x + w) >= (w_img - tol)) | ((y + h) >= (h_img - tol)))


def _drop_border_components(mask):
    """
    Remove ALL connected components touching the image edge (regardless of size).
    Applied to the "raw" mask: a whole edge stain/frame/margins go away in one piece,
    while the inner objects (faults, boundaries) stay untouched.
    """
    h_img, w_img = mask.shape[:2]
    tol = _border_tolerance(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = ~_border_touch_flags(stats, w_img, h_img, tol)
    return _apply_label_keep(labels, keep)


def _keep_line_like(mask):
    """
    Keep only the LONG and THIN connected components (lines), discarding short blobs
    and thick patches. Criterion: the long side of the bbox >= DARK_MIN_LENGTH AND
    the mean thickness (area / long side) <= DARK_MAX_THICKNESS.
    """
    h_img, w_img = mask.shape[:2]
    tol = _border_tolerance(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    w = stats[:, cv2.CC_STAT_WIDTH]
    h = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    long_side = np.maximum(w, h)
    thickness = np.divide(area, long_side, out=np.zeros(len(area), dtype=float),
                          where=long_side > 0)
    keep = (long_side >= config.DARK_MIN_LENGTH) & (thickness <= config.DARK_MAX_THICKNESS)
    if config.DROP_BORDER_TOUCHING:
        keep &= ~_border_touch_flags(stats, w_img, h_img, tol)
    return _apply_label_keep(labels, keep)


def _remove_small_components(mask, min_area):
    """
    Remove connected white regions smaller than min_area pixels (specks, letters),
    as well as edge components (aging stains/frame). Long lines inside stay.
    """
    h_img, w_img = mask.shape[:2]
    tol = _border_tolerance(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    if config.DROP_BORDER_TOUCHING:
        keep &= ~_border_touch_flags(stats, w_img, h_img, tol)
    return _apply_label_keep(labels, keep)
