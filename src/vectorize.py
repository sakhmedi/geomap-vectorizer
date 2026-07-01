"""
vectorize.py — stage 5: turn the clean binary mask into vectors (lists of points).

Idea: cv2.findContours traces each white blob with a polyline (a contour).
Then cv2.approxPolyDP discards excess points — the line becomes smooth and light.

The result is a list of "features": each feature = one line with its points (in pixels),
type (fault/boundary) and source color. This is almost GeoJSON already, just without a file.

The coordinates here are pixel coordinates: (x right, y down). Conversion to WGS84 is done
by a separate georeferencing stage (src/georef.py) over these points, if an AOI is given.
"""

import cv2
import numpy as np

from src import config

# scikit-image is needed only for the centerlines (skeletonization). If it is absent —
# we silently fall back to the contour path (loops), the pipeline does not break.
try:
    from skimage.morphology import skeletonize as _sk_skeletonize
    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    _HAS_SKIMAGE = False


def vectorize(cleaned, prepared, saver):
    """
    cleaned — the dict from cleanup: {"color_masks", "combined", "canny"}.
    prepared — only the color frame is needed, for drawing the overlay.
    Returns a list of features:
      [{"points": [(x, y), ...], "type": "fault", "color": "red", "length_px": float}, ...]
    """
    features = []

    # Vectorize each cleaned color mask.
    for color_name, spec in cleaned["color_masks"].items():
        polylines = _mask_to_polylines(spec["mask"])
        for pts, length in polylines:
            features.append({
                "points": pts,
                "type": spec["type"],
                "color": color_name,
                "length_px": length,
            })

    # If there was no color (tracing paper) — vectorize the Canny edges as type "edge".
    if not cleaned["color_masks"] and cleaned["canny"] is not None:
        polylines = _mask_to_polylines(cleaned["canny"])
        for pts, length in polylines:
            features.append({
                "points": pts,
                "type": "edge",
                "color": "none",
                "length_px": length,
            })

    # The most important debug frame: vectors over the original — check by eye.
    overlay = _draw_overlay(prepared["color"], features)
    saver.save("vectors_overlay", overlay)

    return features


def _mask_to_polylines(mask):
    """
    Turn the white blobs of the mask into polylines.
    If skeletonization is available — trace the CENTERLINES (each point visited once).
    Otherwise — fall back to contours (tracing the blob, points are doubled).
    Returns a list of (points, length_px), where points is a list of (x, y).
    """
    if config.USE_SKELETON and _HAS_SKIMAGE:
        return _skeleton_to_polylines(mask)
    return _contour_to_polylines(mask)


def _contour_to_polylines(mask):
    """Fallback: trace white blobs with contours and simplify (a contour is a closed loop)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        # Contours that are too short are garbage, skip them.
        if len(contour) < config.MIN_CONTOUR_POINTS:
            continue
        # Simplify the polyline (Douglas-Peucker): fewer points, the same shape.
        approx = cv2.approxPolyDP(contour, config.APPROX_EPSILON, True)
        pts = [(int(x), int(y)) for x, y in approx.reshape(-1, 2)]
        if len(pts) < 2:
            continue
        length = float(cv2.arcLength(approx, True))
        result.append((pts, length))
    return result


# The 8 neighbors of a pixel (for traversing the skeleton).
_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_to_polylines(mask):
    """
    Thin the mask down to 1px lines (a skeleton) and trace each branch as a polyline.

    Algorithm: count the number of neighbors of each skeleton pixel. Points with one
    neighbor are line ends, those with three or more are junctions. We walk from each
    end/junction along the neighbors until we hit another end/junction. Closed loops
    without ends are traversed separately. This way each point ends up in the result
    ONCE (not doubled).
    """
    skel = _sk_skeletonize(mask > 0)
    coords = {(int(y), int(x)) for y, x in zip(*np.where(skel))}
    if not coords:
        return []

    neighbors = {p: _skel_neighbors(p, coords) for p in coords}
    visited_edges = set()
    polylines = []

    # Start points — ends (1 neighbor) and junctions (>=3 neighbors).
    nodes = [p for p in coords if len(neighbors[p]) != 2]
    for node in nodes:
        for nb in neighbors[node]:
            if (node, nb) in visited_edges:
                continue
            path = _trace_branch(node, nb, neighbors, visited_edges)
            if len(path) >= 2:
                polylines.append(path)

    # Isolated loops (all points have exactly 2 neighbors, no nodes).
    for p in coords:
        for nb in neighbors[p]:
            if (p, nb) not in visited_edges:
                path = _trace_branch(p, nb, neighbors, visited_edges)
                if len(path) >= 2:
                    polylines.append(path)

    result = []
    for path in polylines:
        # path — a list of (row, col); convert to (x, y) = (col, row).
        pts_xy = [(c, r) for (r, c) in path]
        length = _polyline_length(pts_xy)
        if length < config.MIN_SKELETON_LENGTH:
            continue
        simplified = _simplify(pts_xy)
        if len(simplified) >= 2:
            result.append((simplified, length))
    return result


def _skel_neighbors(p, coords):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in _NEIGHBORS if (r + dr, c + dc) in coords]


def _trace_branch(start, first, neighbors, visited_edges):
    """Walk from start through first along the line until the branch ends/junctions."""
    path = [start]
    prev, cur = start, first
    visited_edges.add((prev, cur))
    visited_edges.add((cur, prev))
    while True:
        path.append(cur)
        nbs = neighbors[cur]
        # An ordinary line point has exactly 2 neighbors — go to the one we didn't come from.
        if len(nbs) != 2:
            break  # reached an end or a junction
        nxt = nbs[0] if nbs[1] == prev else nbs[1]
        if (cur, nxt) in visited_edges:
            break
        visited_edges.add((cur, nxt))
        visited_edges.add((nxt, cur))
        prev, cur = cur, nxt
    return path


def _polyline_length(pts):
    total = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        total += (dx * dx + dy * dy) ** 0.5
    return total


def _simplify(pts_xy):
    """Simplify an open polyline (Douglas-Peucker, closed=False — no doubling)."""
    arr = np.array(pts_xy, dtype=np.int32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(arr, config.APPROX_EPSILON, False)
    return [(int(x), int(y)) for x, y in approx.reshape(-1, 2)]


def _draw_overlay(color_image, features):
    """Draw all vectors over a copy of the color frame (for visual inspection)."""
    overlay = color_image.copy()
    # Outline colors in BGR by feature type.
    type_colors = {
        "fault": (0, 0, 255),      # red
        "boundary": (0, 255, 0),   # green
        "edge": (255, 0, 0),       # blue
    }
    for f in features:
        color = type_colors.get(f["type"], (0, 255, 255))
        pts = f["points"]
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i + 1], color, 2)
    return overlay
