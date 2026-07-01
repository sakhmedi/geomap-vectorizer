"""
georef.py — georeferencing the vectors: map pixels -> WGS84 (latitude/longitude).

The idea, which can be automated without manually placing points:
  1) find the rectangular map frame (neat-line) -> 4 corners in pixels (GCPs);
  2) take the given Area of Interest (AOI) -> 4 corners in ground coordinates;
  3) build a homography pixel -> AOI coordinates (cv2.getPerspectiveTransform);
  4) convert the result to WGS84 via pyproj.

IMPORTANT about the datum (mentor's advice): Soviet maps are Pulkovo 1942, NOT WGS84.
If you treat their coordinates as WGS84, you get a ~100+ m shift. So the AOI's source CRS
defaults to EPSG:4284 (Pulkovo 1942), and we EXPLICITLY transform it to EPSG:4326 (WGS84)
via pyproj. If the AOI is already in WGS84 — specify its EPSG, and the transformation
becomes nearly the identity (but honest).

If the frame is not found or there is no AOI — we return None and the pipeline honestly
stays in pixels (georeferenced=no). Better to hand over pixels than to "reference" at random.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from src import config, crop

# pyproj — the only mandatory new georeferencing dependency.
try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:  # pragma: no cover - environment without pyproj
    _HAS_PYPROJ = False


class GeoTransform:
    """
    A ready referencing for one map: a homography (pixel -> source CRS) + a datum shift
    to WGS84. Call .to_wgs84(points) to convert a list of (x, y) points.
    """

    def __init__(self, homography, source_epsg, gcp_count, rms_px):
        self.homography = homography
        self.source_epsg = source_epsg
        self.gcp_count = gcp_count
        self.rms_px = rms_px
        if not _HAS_PYPROJ:
            raise RuntimeError("pyproj is not installed — georeferencing is unavailable")
        # always_xy=True: input/output are (longitude, latitude), not (latitude, longitude).
        self._to_wgs84 = Transformer.from_crs(
            f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True
        )

    def to_wgs84(self, points):
        """points — a list of (x_px, y_px). Returns a list of (lon, lat) in WGS84."""
        if not points:
            return []
        src = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
        # 1) pixel -> source CRS coordinates (Pulkovo 1942 by default)
        proj = cv2.perspectiveTransform(src, self.homography).reshape(-1, 2)
        # 2) source CRS -> WGS84 (accounting for the datum shift)
        lon, lat = self._to_wgs84.transform(proj[:, 0], proj[:, 1])
        return list(zip(lon.tolist(), lat.tolist()))


# ----------------------------------------------------------------------------
# Step 1: map frame corners in pixels (GCPs)
# ----------------------------------------------------------------------------

def find_corner_gcps(image):
    """
    Find the 4 map frame corners in pixels, ordered TL, TR, BR, BL.
    Reuses the rectangle detector from crop.py. None if there is no frame.
    """
    quad = crop.find_map_corners(image)
    if quad is None:
        return None
    return _order_corners(quad)


def _order_corners(pts):
    """Order 4 points as TL, TR, BR, BL (robustly via sums/differences)."""
    pts = np.array(pts, dtype=np.float64).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


# ----------------------------------------------------------------------------
# Step 2: loading the AOI (ground corners + source CRS)
# ----------------------------------------------------------------------------

def load_aoi(map_name, aoi_path):
    """
    Find and parse the AOI for the map. aoi_path can be:
      - a folder of sidecars named after the map (<map_name>.geojson/.json/.txt),
      - a single file (applied to all maps).
    Returns (corners, source_epsg) or None.
      corners — 4 points [TL, TR, BR, BL] in source CRS coordinates,
      source_epsg — int (default config.DEFAULT_SOURCE_EPSG = 4284, Pulkovo 1942).
    Never crashes: on any read problem it returns None.
    """
    if not aoi_path:
        return None
    path = Path(aoi_path)
    sidecar = _resolve_aoi_file(path, map_name)
    if sidecar is None:
        return None
    try:
        if sidecar.suffix.lower() in (".geojson", ".json"):
            return _parse_aoi_geojson(sidecar)
        if sidecar.suffix.lower() == ".txt":
            return _parse_aoi_txt(sidecar)
    except Exception:
        return None
    return None


def _resolve_aoi_file(path, map_name):
    """Pick a specific AOI file: a sidecar named after the map or a single shared file."""
    if path.is_file():
        return path
    if path.is_dir():
        for ext in (".geojson", ".json", ".txt"):
            candidate = path / f"{map_name}{ext}"
            if candidate.is_file():
                return candidate
        # A single AOI for the whole dataset, if one file was placed in the folder.
        files = sorted(p for p in path.iterdir()
                       if p.suffix.lower() in (".geojson", ".json", ".txt"))
        if len(files) == 1:
            return files[0]
    return None


def _parse_aoi_geojson(path):
    """
    Extract 4 corners from GeoJSON. Supports Polygon/Feature/FeatureCollection.
    The CRS is taken from the 'crs' field (EPSG), otherwise the default (Pulkovo 1942).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    epsg = _epsg_from_geojson(data)
    coords = _first_polygon_ring(data)
    if coords is None:
        return None
    corners = _corners_from_ring(coords)
    return corners, epsg


def _epsg_from_geojson(data):
    crs = data.get("crs") if isinstance(data, dict) else None
    if isinstance(crs, dict):
        name = crs.get("properties", {}).get("name", "")
        digits = "".join(ch for ch in str(name) if ch.isdigit())
        if digits:
            return int(digits)
    return config.DEFAULT_SOURCE_EPSG


def _first_polygon_ring(data):
    """Return the outer ring of the first polygon found as a list of [x, y]."""
    if data.get("type") == "FeatureCollection":
        for feat in data.get("features", []):
            ring = _ring_from_geometry(feat.get("geometry"))
            if ring:
                return ring
        return None
    if data.get("type") == "Feature":
        return _ring_from_geometry(data.get("geometry"))
    return _ring_from_geometry(data)


def _ring_from_geometry(geom):
    if not isinstance(geom, dict):
        return None
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Polygon" and coords:
        return coords[0]
    if gtype == "MultiPolygon" and coords:
        return coords[0][0]
    return None


def _parse_aoi_txt(path):
    """
    A text AOI. We support:
      - a first line '# epsg=4284' (optional),
      - either 4 lines of 'lon lat' (corners clockwise/in any order — we'll order them),
      - or a single line 'minlon minlat maxlon maxlat' (bbox).
    """
    epsg = config.DEFAULT_SOURCE_EPSG
    points = []
    bbox = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("#") or low.startswith("epsg"):
            digits = "".join(ch for ch in low if ch.isdigit())
            if digits:
                epsg = int(digits)
            continue
        nums = [float(x) for x in line.replace(",", " ").split()]
        if len(nums) == 4 and not points:
            bbox = nums  # minlon minlat maxlon maxlat
        elif len(nums) >= 2:
            points.append((nums[0], nums[1]))
    if bbox is not None:
        corners = _corners_from_bbox(*bbox)
        return corners, epsg
    if len(points) >= 4:
        corners = _corners_from_ring(points)
        return corners, epsg
    return None


def _corners_from_ring(ring):
    """
    Make 4 corners [TL, TR, BR, BL] from a ring of coordinates.
    If the ring has exactly 4 unique points — we order them; otherwise we take the bbox.
    """
    pts = [(float(p[0]), float(p[1])) for p in ring]
    # Remove the closing point if the ring is closed.
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) == 4:
        return _order_corners_geo(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return _corners_from_bbox(min(xs), min(ys), max(xs), max(ys))


def _corners_from_bbox(min_lon, min_lat, max_lon, max_lat):
    """bbox -> [TL, TR, BR, BL] for north-up (top = max_lat)."""
    return np.array([
        [min_lon, max_lat],  # TL
        [max_lon, max_lat],  # TR
        [max_lon, min_lat],  # BR
        [min_lon, min_lat],  # BL
    ], dtype=np.float32)


def _order_corners_geo(pts):
    """Order 4 geo-points as TL, TR, BR, BL (top = larger latitude)."""
    pts = np.array(pts, dtype=np.float64)
    # The top pair is the two points with the largest latitude (y), the bottom — the smallest.
    order = pts[np.argsort(pts[:, 1])]  # ascending latitude
    bottom = order[:2]
    top = order[2:]
    tl, tr = top[np.argsort(top[:, 0])]      # left to right by longitude
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


# ----------------------------------------------------------------------------
# Step 3: building the transformation
# ----------------------------------------------------------------------------

def build_transform(pixel_corners, aoi):
    """
    pixel_corners — the 4 frame corners in pixels [TL, TR, BR, BL] (from find_corner_gcps).
    aoi — (corners_geo, source_epsg) from load_aoi.
    Returns a GeoTransform or None.
    """
    if pixel_corners is None or aoi is None or not _HAS_PYPROJ:
        return None
    corners_geo, source_epsg = aoi
    src = np.array(pixel_corners, dtype=np.float32).reshape(4, 2)
    dst = np.array(corners_geo, dtype=np.float32).reshape(4, 2)
    homography = cv2.getPerspectiveTransform(src, dst)
    rms_px = _reprojection_rms(homography, src, dst)
    return GeoTransform(homography, source_epsg, gcp_count=4, rms_px=rms_px)


def _reprojection_rms(homography, src, dst):
    """
    Back-projection residual in pixels: convert dst back to pixels via H^-1 and
    compare with src. For a 4-point homography it is ~0, but we compute it honestly
    (it catches degenerate/collinear cases).
    """
    try:
        inv = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return float("nan")
    back = cv2.perspectiveTransform(dst.reshape(-1, 1, 2).astype(np.float64),
                                    inv).reshape(-1, 2)
    diff = back - src.astype(np.float64)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def georeference(image, map_name, aoi_path):
    """
    A convenience wrapper for the pipeline: run steps 1-3.
    Returns (GeoTransform | None, info_dict) — info for the summary/logs.
    """
    info = {"georeferenced": False, "reason": ""}
    if not aoi_path:
        info["reason"] = "AOI not given (--aoi)"
        return None, info
    if not _HAS_PYPROJ:
        info["reason"] = "pyproj is not installed"
        return None, info

    pixel_corners = find_corner_gcps(image)
    if pixel_corners is None:
        info["reason"] = "map frame not found"
        return None, info

    aoi = load_aoi(map_name, aoi_path)
    if aoi is None:
        info["reason"] = "AOI for the map not found/not recognized"
        return None, info

    transform = build_transform(pixel_corners, aoi)
    if transform is None:
        info["reason"] = "could not build the transformation"
        return None, info

    info.update({
        "georeferenced": True,
        "source_epsg": transform.source_epsg,
        "gcp_count": transform.gcp_count,
        "rms_px": transform.rms_px,
    })
    return transform, info
