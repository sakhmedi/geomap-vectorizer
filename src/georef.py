"""
georef.py — геопривязка вектора: пиксели карты -> WGS84 (широта/долгота).

Идея, которую можно автоматизировать без ручной расстановки точек:
  1) находим прямоугольную рамку карты (neat-line) -> 4 угла в пикселях (GCP);
  2) берём заданную Area of Interest (AOI) -> 4 угла в координатах местности;
  3) строим гомографию пиксель -> координаты AOI (cv2.getPerspectiveTransform);
  4) переводим результат в WGS84 через pyproj.

ВАЖНО про датум (совет ментора): советские карты — это Пулково 1942, а НЕ WGS84.
Если воспринять их координаты как WGS84, получишь сдвиг ~100+ м. Поэтому исходный CRS
AOI по умолчанию EPSG:4284 (Пулково 1942), и мы ЯВНО трансформируем его в EPSG:4326
(WGS84) через pyproj. Если AOI уже в WGS84 — укажите его EPSG, и трансформация станет
почти тождественной (но честной).

Если рамка не найдена или AOI нет — возвращаем None и пайплайн честно остаётся в
пикселях (georeferenced=no). Лучше отдать пиксели, чем «привязать» наугад.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from src import config, crop

# pyproj — единственная обязательная новая зависимость геопривязки.
try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:  # pragma: no cover - окружение без pyproj
    _HAS_PYPROJ = False


class GeoTransform:
    """
    Готовая привязка одной карты: гомография (пиксель -> исходный CRS) + датум-переход
    в WGS84. Зовите .to_wgs84(points) для перевода списка точек (x, y).
    """

    def __init__(self, homography, source_epsg, gcp_count, rms_px):
        self.homography = homography
        self.source_epsg = source_epsg
        self.gcp_count = gcp_count
        self.rms_px = rms_px
        if not _HAS_PYPROJ:
            raise RuntimeError("pyproj не установлен — геопривязка недоступна")
        # always_xy=True: на вход/выход (долгота, широта), а не (широта, долгота).
        self._to_wgs84 = Transformer.from_crs(
            f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True
        )

    def to_wgs84(self, points):
        """points — список (x_px, y_px). Возвращает список (lon, lat) в WGS84."""
        if not points:
            return []
        src = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
        # 1) пиксель -> координаты исходного CRS (Пулково 1942 по умолчанию)
        proj = cv2.perspectiveTransform(src, self.homography).reshape(-1, 2)
        # 2) исходный CRS -> WGS84 (учёт сдвига датума)
        lon, lat = self._to_wgs84.transform(proj[:, 0], proj[:, 1])
        return list(zip(lon.tolist(), lat.tolist()))


# ----------------------------------------------------------------------------
# Шаг 1: углы рамки карты в пикселях (GCP)
# ----------------------------------------------------------------------------

def find_corner_gcps(image):
    """
    Найти 4 угла рамки карты в пикселях, упорядоченные TL, TR, BR, BL.
    Переиспользует детектор прямоугольника из crop.py. None, если рамки нет.
    """
    quad = crop.find_map_corners(image)
    if quad is None:
        return None
    return _order_corners(quad)


def _order_corners(pts):
    """Упорядочить 4 точки как TL, TR, BR, BL (надёжно через суммы/разности)."""
    pts = np.array(pts, dtype=np.float64).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


# ----------------------------------------------------------------------------
# Шаг 2: загрузка AOI (углы местности + исходный CRS)
# ----------------------------------------------------------------------------

def load_aoi(map_name, aoi_path):
    """
    Найти и распарсить AOI для карты. aoi_path может быть:
      - папкой с сайдкарами по имени карты (<map_name>.geojson/.json/.txt),
      - одним файлом (применяется ко всем картам).
    Возвращает (corners, source_epsg) или None.
      corners — 4 точки [TL, TR, BR, BL] в координатах исходного CRS,
      source_epsg — int (по умолчанию config.DEFAULT_SOURCE_EPSG = 4284, Пулково 1942).
    Никогда не падает: при любой проблеме чтения возвращает None.
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
    """Выбрать конкретный AOI-файл: сайдкар по имени карты или единый файл."""
    if path.is_file():
        return path
    if path.is_dir():
        for ext in (".geojson", ".json", ".txt"):
            candidate = path / f"{map_name}{ext}"
            if candidate.is_file():
                return candidate
        # Единый AOI на весь датасет, если положили один файл в папку.
        files = sorted(p for p in path.iterdir()
                       if p.suffix.lower() in (".geojson", ".json", ".txt"))
        if len(files) == 1:
            return files[0]
    return None


def _parse_aoi_geojson(path):
    """
    Достать 4 угла из GeoJSON. Поддержка Polygon/Feature/FeatureCollection.
    CRS берём из поля 'crs' (EPSG), иначе — дефолт (Пулково 1942).
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
    """Вернуть внешнее кольцо первого попавшегося полигона как список [x, y]."""
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
    Текстовый AOI. Поддерживаем:
      - первую строку '# epsg=4284' (необязательно),
      - либо 4 строки 'lon lat' (углы по часовой/в любом порядке — упорядочим),
      - либо одну строку 'minlon minlat maxlon maxlat' (bbox).
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
    Из кольца координат сделать 4 угла [TL, TR, BR, BL].
    Если в кольце ровно 4 уникальные точки — упорядочиваем их; иначе берём bbox.
    """
    pts = [(float(p[0]), float(p[1])) for p in ring]
    # Убираем замыкающую точку, если кольцо закрыто.
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) == 4:
        return _order_corners_geo(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return _corners_from_bbox(min(xs), min(ys), max(xs), max(ys))


def _corners_from_bbox(min_lon, min_lat, max_lon, max_lat):
    """bbox -> [TL, TR, BR, BL] для north-up (верх = max_lat)."""
    return np.array([
        [min_lon, max_lat],  # TL
        [max_lon, max_lat],  # TR
        [max_lon, min_lat],  # BR
        [min_lon, min_lat],  # BL
    ], dtype=np.float32)


def _order_corners_geo(pts):
    """Упорядочить 4 гео-точки как TL, TR, BR, BL (верх = большая широта)."""
    pts = np.array(pts, dtype=np.float64)
    # Верхняя пара — две точки с наибольшей широтой (y), нижняя — с наименьшей.
    order = pts[np.argsort(pts[:, 1])]  # по возрастанию широты
    bottom = order[:2]
    top = order[2:]
    tl, tr = top[np.argsort(top[:, 0])]      # слева направо по долготе
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


# ----------------------------------------------------------------------------
# Шаг 3: построение трансформации
# ----------------------------------------------------------------------------

def build_transform(pixel_corners, aoi):
    """
    pixel_corners — 4 угла рамки в пикселях [TL, TR, BR, BL] (из find_corner_gcps).
    aoi — (corners_geo, source_epsg) из load_aoi.
    Возвращает GeoTransform или None.
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
    Невязка обратной проекции в пикселях: переводим dst обратно в пиксели через
    H^-1 и сравниваем со src. Для 4-точечной гомографии ~0, но честно считаем
    (ловит вырожденные/коллинеарные случаи).
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
    Удобная обёртка для пайплайна: пройти шаги 1-3.
    Возвращает (GeoTransform | None, info_dict) — info для сводки/логов.
    """
    info = {"georeferenced": False, "reason": ""}
    if not aoi_path:
        info["reason"] = "AOI не задан (--aoi)"
        return None, info
    if not _HAS_PYPROJ:
        info["reason"] = "pyproj не установлен"
        return None, info

    pixel_corners = find_corner_gcps(image)
    if pixel_corners is None:
        info["reason"] = "рамка карты не найдена"
        return None, info

    aoi = load_aoi(map_name, aoi_path)
    if aoi is None:
        info["reason"] = "AOI для карты не найден/не распознан"
        return None, info

    transform = build_transform(pixel_corners, aoi)
    if transform is None:
        info["reason"] = "не удалось построить трансформацию"
        return None, info

    info.update({
        "georeferenced": True,
        "source_epsg": transform.source_epsg,
        "gcp_count": transform.gcp_count,
        "rms_px": transform.rms_px,
    })
    return transform, info
