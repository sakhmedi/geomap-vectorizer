"""
legend.py — map legend extraction (Track 2: "legend extraction").

A legend is a table "color swatch -> geological layer name". The previous stage
(extract) deliberately CUTS the legend off with the frame mask, so that its colored
swatches do not turn into false vectors. Here we do the opposite: in a separate pass
we find the colored SWATCHES in the legend and link each one to a feature class
(fault/boundary) and to the already extracted vectors of the same color.

How we tell a legend swatch from a fault line (both may be, say, red):
  - a legend swatch is a DENSE compact fill (area / area(bbox) is high),
  - a fault line is thin and winding (this ratio is low).
Plus swatches usually lie OUTSIDE the map frame (on the margins) — that is the second filter.

OCR of the layer labels (Cyrillic) is a Track 1 task; here we skip OCR to stay
lightweight and reproducible out of the box. Result: a list of swatches with their color,
class, bbox and the number of linked vectors on the map.
"""

import cv2
import numpy as np

from src import config, crop, extract


def extract_legend(color_image, profile_name, features=None, saver=None):
    """
    Find the legend swatches on the map.

    color_image — the color (BGR) frame (the same one that goes into HSV extraction).
    profile_name — the color profile from config.PROFILES.
    features — the list of vectors from vectorize (to link "swatch -> how many lines").
    saver — DebugSaver (we'll draw the found swatches for eyeballing).

    Returns (entries, summary):
      entries — a list of swatches: {color, type, bbox_px, mean_hsv, area_px, outside_frame}
      summary — a per-color summary: [{color, type, num_swatches, num_map_features}]
    Never crashes: on any problem it returns empty lists.
    """
    if not config.EXTRACT_LEGEND:
        return [], []
    profile = config.PROFILES.get(profile_name, {})
    if not profile:
        return [], []

    try:
        frame = crop.find_map_corners(color_image)
    except Exception:
        frame = None

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    h_img, w_img = color_image.shape[:2]
    img_area = float(h_img * w_img)

    entries = []
    for color_name, spec in profile.items():
        mask = extract.color_mask(hsv, spec["ranges"])
        entries.extend(
            _swatches_from_mask(mask, hsv, color_name, spec["type"], frame, img_area)
        )

    summary = _summarize(entries, profile, features or [])

    if saver is not None:
        saver.save("legend_swatches", _draw_overlay(color_image, entries))

    return entries, summary


def _swatches_from_mask(mask, hsv, color_name, color_type, frame, img_area):
    """Pull the components that look like legend swatches out of one color mask."""
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for i in range(1, num):  # 0 is the background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < config.LEGEND_MIN_SWATCH_AREA_FRAC * img_area:
            continue
        if area > config.LEGEND_MAX_SWATCH_AREA_FRAC * img_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        bbox_area = w * h
        if bbox_area == 0:
            continue
        # Fill density: a legend swatch is filled almost entirely, a line is not.
        if area / bbox_area < config.LEGEND_MIN_FILL_RATIO:
            continue
        aspect = w / h if h else 0.0
        if not (config.LEGEND_ASPECT_MIN <= aspect <= config.LEGEND_ASPECT_MAX):
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        outside_frame = _is_outside_frame(frame, cx, cy)
        if config.LEGEND_REQUIRE_OUTSIDE_FRAME and frame is not None and not outside_frame:
            continue
        mean_hsv = _mean_hsv(hsv, labels, i)
        found.append({
            "color": color_name,
            "type": color_type,
            "bbox_px": [x, y, w, h],
            "mean_hsv": mean_hsv,
            "area_px": area,
            "outside_frame": bool(outside_frame),
        })
    return found


def _is_outside_frame(frame, cx, cy):
    """True if the point lies OUTSIDE the frame quadrilateral. Without a frame — treat as True."""
    if frame is None:
        return True
    quad = np.asarray(frame, dtype=np.float32).reshape(-1, 1, 2)
    # >0 inside, <0 outside, =0 on the boundary.
    return cv2.pointPolygonTest(quad, (cx, cy), False) < 0


def _mean_hsv(hsv, labels, label_id):
    """The mean HSV over the component's pixels (for reporting/debugging the thresholds)."""
    sel = labels == label_id
    if not np.any(sel):
        return [0, 0, 0]
    vals = hsv[sel].mean(axis=0)
    return [int(round(v)) for v in vals]


def _summarize(entries, profile, features):
    """Per-color summary: how many swatches and how many vectors of the same color on the map."""
    feat_counts = {}
    for f in features:
        feat_counts[f.get("color")] = feat_counts.get(f.get("color"), 0) + 1

    summary = []
    for color_name, spec in profile.items():
        swatches = [e for e in entries if e["color"] == color_name]
        if not swatches:
            continue
        summary.append({
            "color": color_name,
            "type": spec["type"],
            "num_swatches": len(swatches),
            "num_map_features": feat_counts.get(color_name, 0),
        })
    return summary


def _draw_overlay(color_image, entries):
    """Box the found legend swatches (for visual inspection in debug/)."""
    overlay = color_image.copy()
    for e in entries:
        x, y, w, h = e["bbox_px"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(overlay, e["color"], (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay
