"""
vectorize.py — этап 5: из чистой бинарной маски делаем векторы (списки точек).

Идея: cv2.findContours обводит каждое белое пятно ломаной линией (контуром).
Затем cv2.approxPolyDP выкидывает лишние точки — линия становится гладкой и лёгкой.

Результат — список «фич»: каждая фича = одна линия с её точками (в пикселях),
типом (fault/boundary) и цветом-источником. Это уже почти GeoJSON, только без файла.

Координаты здесь пиксельные: (x вправо, y вниз). Перевод в WGS84 делает отдельный
этап геопривязки (src/georef.py) уже над этими точками, если задан AOI.
"""

import cv2
import numpy as np

from src import config

# scikit-image нужен только для осевых линий (скелетизация). Если его нет —
# тихо откатываемся на контурный путь (петли), пайплайн не ломается.
try:
    from skimage.morphology import skeletonize as _sk_skeletonize
    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    _HAS_SKIMAGE = False


def vectorize(cleaned, prepared, saver):
    """
    cleaned — словарь из cleanup: {"color_masks", "combined", "canny"}.
    prepared — нужен только цветной кадр для рисования overlay.
    Возвращает список фич:
      [{"points": [(x, y), ...], "type": "fault", "color": "red", "length_px": float}, ...]
    """
    features = []

    # Векторизуем каждую очищенную цветную маску.
    for color_name, spec in cleaned["color_masks"].items():
        polylines = _mask_to_polylines(spec["mask"])
        for pts, length in polylines:
            features.append({
                "points": pts,
                "type": spec["type"],
                "color": color_name,
                "length_px": length,
            })

    # Если цвета не было (калька) — векторизуем края Canny как тип "edge".
    if not cleaned["color_masks"] and cleaned["canny"] is not None:
        polylines = _mask_to_polylines(cleaned["canny"])
        for pts, length in polylines:
            features.append({
                "points": pts,
                "type": "edge",
                "color": "none",
                "length_px": length,
            })

    # Самый важный debug-кадр: вектора поверх оригинала — проверяем глазами.
    overlay = _draw_overlay(prepared["color"], features)
    saver.save("vectors_overlay", overlay)

    return features


def _mask_to_polylines(mask):
    """
    Превратить белые пятна маски в полилинии.
    Если доступна скелетизация — трассируем ОСЕВЫЕ линии (точка проходится один раз).
    Иначе — фолбэк на контуры (обводка пятна, точки задваиваются).
    Возвращает список (points, length_px), где points — список (x, y).
    """
    if config.USE_SKELETON and _HAS_SKIMAGE:
        return _skeleton_to_polylines(mask)
    return _contour_to_polylines(mask)


def _contour_to_polylines(mask):
    """Фолбэк: обвести белые пятна контурами и упростить (контур — замкнутая петля)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        # Слишком короткие контуры — мусор, пропускаем.
        if len(contour) < config.MIN_CONTOUR_POINTS:
            continue
        # Упрощаем ломаную (Douglas-Peucker): меньше точек, та же форма.
        approx = cv2.approxPolyDP(contour, config.APPROX_EPSILON, True)
        pts = [(int(x), int(y)) for x, y in approx.reshape(-1, 2)]
        if len(pts) < 2:
            continue
        length = float(cv2.arcLength(approx, True))
        result.append((pts, length))
    return result


# 8 соседей пикселя (для обхода скелета).
_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_to_polylines(mask):
    """
    Утончить маску до линий в 1px (скелет) и проследить каждую ветку как полилинию.

    Алгоритм: считаем число соседей у каждого скелетного пикселя. Точки с одним
    соседом — концы линий, с тремя и более — развилки. Идём от каждого конца/развилки
    вдоль соседей, пока не упрёмся в другой конец/развилку. Замкнутые петли без концов
    обходим отдельно. Так каждая точка попадает в результат ОДИН раз (не задваивается).
    """
    skel = _sk_skeletonize(mask > 0)
    coords = {(int(y), int(x)) for y, x in zip(*np.where(skel))}
    if not coords:
        return []

    neighbors = {p: _skel_neighbors(p, coords) for p in coords}
    visited_edges = set()
    polylines = []

    # Стартовые точки — концы (1 сосед) и развилки (>=3 соседа).
    nodes = [p for p in coords if len(neighbors[p]) != 2]
    for node in nodes:
        for nb in neighbors[node]:
            if (node, nb) in visited_edges:
                continue
            path = _trace_branch(node, nb, neighbors, visited_edges)
            if len(path) >= 2:
                polylines.append(path)

    # Изолированные петли (все точки имеют по 2 соседа, узлов нет).
    for p in coords:
        for nb in neighbors[p]:
            if (p, nb) not in visited_edges:
                path = _trace_branch(p, nb, neighbors, visited_edges)
                if len(path) >= 2:
                    polylines.append(path)

    result = []
    for path in polylines:
        # path — список (row, col); переводим в (x, y) = (col, row).
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
    """Пройти от start через first вдоль линии, пока ветка не кончится/не развилка."""
    path = [start]
    prev, cur = start, first
    visited_edges.add((prev, cur))
    visited_edges.add((cur, prev))
    while True:
        path.append(cur)
        nbs = neighbors[cur]
        # На обычной точке линии ровно 2 соседа — идём в тот, откуда не пришли.
        if len(nbs) != 2:
            break  # дошли до конца или развилки
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
    """Упростить открытую ломаную (Douglas-Peucker, closed=False — без задвоения)."""
    arr = np.array(pts_xy, dtype=np.int32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(arr, config.APPROX_EPSILON, False)
    return [(int(x), int(y)) for x, y in approx.reshape(-1, 2)]


def _draw_overlay(color_image, features):
    """Нарисовать все вектора поверх копии цветного кадра (для визуальной проверки)."""
    overlay = color_image.copy()
    # Цвета обводки в BGR по типу объекта.
    type_colors = {
        "fault": (0, 0, 255),      # красный
        "boundary": (0, 255, 0),   # зелёный
        "edge": (255, 0, 0),       # синий
    }
    for f in features:
        color = type_colors.get(f["type"], (0, 255, 255))
        pts = f["points"]
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i + 1], color, 2)
    return overlay
