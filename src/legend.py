"""
legend.py — извлечение легенды карты (Трек 2: «извлечение легенды»).

Легенда — это таблица «образец цвета -> название геологического слоя». Прошлый
этап (extract) намеренно ОТСЕКАЕТ легенду маской по рамке, чтобы её цветные
образцы не превращались в ложные вектора. Здесь мы делаем обратное: отдельным
проходом находим в легенде цветные ОБРАЗЦЫ и связываем каждый с классом объекта
(fault/boundary) и с уже извлечёнными векторами того же цвета.

Как отличаем образец легенды от линии разлома (оба, например, красные):
  - образец легенды — ПЛОТНАЯ компактная заливка (area / area(bbox) высокое),
  - линия разлома — тонкая и извилистая (это отношение низкое).
Плюс образцы обычно лежат ВНЕ рамки карты (на полях) — это второй фильтр.

OCR подписей слоёв (кириллица) — задача Трека 1; здесь без OCR, чтобы остаться
лёгким и воспроизводимым «из коробки». Результат: список образцов с их цветом,
классом, bbox и числом связанных векторов на карте.
"""

import cv2
import numpy as np

from src import config, crop, extract


def extract_legend(color_image, profile_name, features=None, saver=None):
    """
    Найти образцы легенды на карте.

    color_image — цветной (BGR) кадр (тот же, что идёт в HSV-извлечение).
    profile_name — профиль цветов из config.PROFILES.
    features — список векторов из vectorize (для связи «образец -> сколько линий»).
    saver — DebugSaver (нарисуем найденные образцы для проверки глазами).

    Возвращает (entries, summary):
      entries — список образцов: {color, type, bbox_px, mean_hsv, area_px, outside_frame}
      summary — сводка по цветам: [{color, type, num_swatches, num_map_features}]
    Никогда не падает: при любой проблеме возвращает пустые списки.
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
    """Вытащить из одной цветовой маски компоненты, похожие на образцы легенды."""
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for i in range(1, num):  # 0 — фон
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
        # Плотность заливки: образец легенды залит почти целиком, линия — нет.
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
    """True, если точка лежит ВНЕ четырёхугольника рамки. Без рамки — считаем True."""
    if frame is None:
        return True
    quad = np.asarray(frame, dtype=np.float32).reshape(-1, 1, 2)
    # >0 внутри, <0 снаружи, =0 на границе.
    return cv2.pointPolygonTest(quad, (cx, cy), False) < 0


def _mean_hsv(hsv, labels, label_id):
    """Средний HSV по пикселям компоненты (для отчёта/отладки порогов)."""
    sel = labels == label_id
    if not np.any(sel):
        return [0, 0, 0]
    vals = hsv[sel].mean(axis=0)
    return [int(round(v)) for v in vals]


def _summarize(entries, profile, features):
    """Сводка по цветам: сколько образцов и сколько векторов того же цвета на карте."""
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
    """Обвести найденные образцы легенды (для визуальной проверки в debug/)."""
    overlay = color_image.copy()
    for e in entries:
        x, y, w, h = e["bbox_px"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(overlay, e["color"], (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay
