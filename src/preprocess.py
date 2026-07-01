"""
preprocess.py — stage 2: preparing the scan before feature extraction.

The maps are faded and noisy, so they must be "tidied up" before analysis.
On output we return TWO variants of the same map:
  - color : the color version (resized + enhanced contrast) — goes into HSV (stage 3),
  - gray  : the gray version (smoothed) — goes into Canny edge detection (stage 3).

Each step saves a debug frame, so the effect can be seen by eye.
"""

import cv2

from src import config, crop


def preprocess(image, saver):
    """
    Take the source color frame (BGR), return a dict:
        {"color": <BGR for HSV>, "gray": <gray for edges>}
    saver — DebugSaver, the intermediate frames 01..03 go here.
    """
    # --- Step 2.1: shrink large scans ---
    # Fewer pixels = faster processing and less fine noise.
    resized = _resize_max_side(image, config.MAX_IMAGE_SIDE)
    saver.save("resized", resized)

    # --- Step 2.1b: crop to the map frame (safe if found) ---
    cropped_img, crop_offset, cropped = crop.crop_to_map(resized, saver)

    # --- Step 2.2: enhance contrast (CLAHE) ---
    # Faded colors become brighter/more saturated, easier for HSV thresholds to catch.
    # We apply CLAHE to the brightness (the L channel of the LAB space) so as NOT to
    # spoil the colors themselves (the hue stays, only the brightness contrast changes).
    enhanced = _apply_clahe_color(cropped_img)
    saver.save("clahe", enhanced)

    # --- Step 2.3: gray + denoise ---
    # Gray is needed for edge detection. Denoising removes paper grain and faint pencil,
    # so that Canny does not mistake them for "edges".
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    denoised = _denoise_gray(gray, config.DENOISE_STRENGTH)
    saver.save("denoised", denoised)

    return {
        "color": enhanced,
        "gray": denoised,
        "crop_offset": crop_offset,   # (x0, y0) crop offset in resized coordinates
        "cropped": cropped,           # True if we actually cropped
    }


def _resize_max_side(image, max_side):
    """Shrink so the long side is no more than max_side. 0 = leave as is."""
    if not max_side:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _apply_clahe_color(image):
    """Enhance brightness contrast while keeping the colors (via the LAB space)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _denoise_gray(gray, strength):
    """Smooth the gray frame with a median filter. strength=0 -> leave as is."""
    if not strength or strength < 3:
        return gray
    ksize = strength if strength % 2 == 1 else strength + 1  # the kernel must be odd
    return cv2.medianBlur(gray, ksize)
