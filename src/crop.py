"""
crop.py — an optional stage: cropping to the map area (by the neat-line frame).

Why: on scans there are margins and a legend with color swatches around the map.
The swatches in the legend cause false positives (a red/green "icon" becomes a vector).
If we find the rectangular map frame and crop to it — the margins and legend go away.

IMPORTANT — safe fallback: the frame is not always found (faded/frameless maps).
If there is no confident rectangle, we crop NOTHING and return the frame as is.
Better not to crop than to accidentally cut off a piece of the map.
"""

import cv2

from src import config


def crop_to_map(image, saver):
    """
    Try to crop image to the map frame.
    Returns (cropped_image, (x0, y0), cropped_flag):
      - cropped_image — the cropped or the original frame,
      - (x0, y0) — the offset of the top-left corner of the crop (0,0 if not cropped),
      - cropped_flag — True if we actually cropped.
    """
    if not config.CROP_TO_MAP_BORDER:
        return image, (0, 0), False

    rect = _find_map_rectangle(image)
    if rect is None:
        # Frame not found — safely leave it alone.
        return image, (0, 0), False

    x, y, w, h = rect
    cropped = image[y:y + h, x:x + w]
    saver.save("cropped", cropped)
    return cropped, (x, y), True


def find_map_corners(image):
    """
    Find the 4 corners of the rectangular map frame (as an Nx2 array of points in pixels),
    or None if there is no confident quadrilateral.

    This is a public function: georef.py uses it to build GCPs, regardless of whether
    cropping is enabled (CROP_TO_MAP_BORDER). Cropping itself is a separate decision,
    and the frame corners are useful for georeferencing in any case.
    """
    h_img, w_img = image.shape[:2]
    img_area = h_img * w_img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # Thicken the edges so a broken frame joins into a closed contour.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_score = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        # The frame must occupy a noticeable, but not the entire, area of the image.
        if area < config.CROP_MIN_AREA_FRAC * img_area:
            continue
        if area > config.CROP_MAX_AREA_FRAC * img_area:
            continue
        peri = cv2.arcLength(c, True)
        # Creased historical maps rarely give a clean quad with a single epsilon —
        # we try several simplification levels and take the first one that converges
        # to a convex quad.
        approx = None
        for eps in (0.02, 0.03, 0.05, 0.08):
            cand = cv2.approxPolyDP(c, eps * peri, True)
            if len(cand) == 4 and cv2.isContourConvex(cand):
                approx = cand
                break
        if approx is None:
            continue
        # Among the candidates we prefer the most "rectangular" one: the contour area
        # divided by the area of its bounding box is close to 1 for a real frame.
        x, y, w, h = cv2.boundingRect(approx)
        rectangularity = area / float(w * h) if w * h else 0.0
        score = rectangularity * area  # rectangular AND large
        if score > best_score:
            best_score = score
            best_quad = approx.reshape(4, 2)

    return best_quad


def _find_map_rectangle(image):
    """
    Find a large ~rectangular contour (the map frame).
    Returns the (x, y, w, h) bounding box or None if there is no confident frame.
    """
    quad = find_map_corners(image)
    if quad is None:
        return None
    return cv2.boundingRect(quad.astype("int32"))
