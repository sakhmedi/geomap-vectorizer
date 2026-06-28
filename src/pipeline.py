"""
pipeline.py — обработка ОДНОЙ карты от начала до конца.

Этапы: загрузка -> предобработка -> выделение объектов (HSV/тёмные линии/Canny) ->
очистка масок -> векторизация (осевые линии) -> геопривязка (пиксели -> WGS84) ->
экспорт (GeoJSON + Shapefile). Каждый этап сохраняет свой debug-кадр.

Функция возвращает небольшой словарь-отчёт по карте (для сводки _summary.csv).
Принцип: одна плохая карта не роняет весь прогон — помечаем status/confidence/georeferenced.
"""

import cv2

from src import (cleanup, config, export, extract, georef, io_utils, legend,
                 preprocess, vectorize)


def process_map(image_path, input_dir, output_dir, debug_root, profile_name,
                debug_enabled=True, aoi_path=None, use_sam=False):
    """
    Прогнать одну карту через пайплайн. Возвращает dict с результатом:
      {name, status, ...}
    status:
      - "failed"  : файл не прочитался (битый/не картинка)
      - "ok"      : карта прошла весь конвейер (вектора + экспорт); качество
                    отражают поля confidence/georeferenced, а не status
    """
    map_name = io_utils.get_map_name(image_path, input_dir)

    # --- Этап 1: загрузка ---
    image = io_utils.load_image(image_path)
    if image is None:
        # Не падаем на одной плохой карте — помечаем и идём дальше.
        return {"name": map_name, "status": "failed", "reason": "не удалось прочитать файл"}

    saver = io_utils.DebugSaver(debug_root, map_name, enabled=debug_enabled)
    saver.save("original", image)

    height, width = image.shape[:2]

    # --- Этап 2: предобработка ---
    prepared = preprocess.preprocess(image, saver)
    # prepared["color"] -> для HSV, prepared["gray"] -> для краёв.

    # --- Этап 3: выделение объектов (HSV + Canny, опц. SAM) ---
    extracted = extract.extract(prepared, profile_name, saver, use_sam=use_sam)
    # extracted["color_masks"], extracted["canny"], extracted["combined"]

    # --- Этап 4: очистка масок (морфология + фильтр мелочи) ---
    cleaned = cleanup.cleanup(extracted, saver)
    # cleaned["color_masks"], cleaned["combined"], cleaned["canny"]

    # --- Этап 5: векторизация (контуры -> полилинии) ---
    features = vectorize.vectorize(cleaned, prepared, saver)

    # --- Этап 5a: извлечение легенды (образцы цвета -> класс слоя) ---
    legend_entries, legend_summary = legend.extract_legend(
        prepared["color"], profile_name, features=features, saver=saver)

    # --- Триаж: насколько уверены в результате ---
    confidence, reason = _assess_confidence(cleaned["combined"], features)

    # --- Этап 5b: геопривязка (пиксели -> WGS84), если задан AOI ---
    # Привязку строим в той же системе пикселей, что и вектора (prepared["color"]).
    geo_transform, geo_info = georef.georeference(prepared["color"], map_name, aoi_path)

    # --- Этап 6: экспорт в GeoJSON (+ опц. Shapefile) ---
    # prepared["color"] мог быть ужат — берём его размер, чтобы координаты совпадали с векторами.
    out_h, out_w = prepared["color"].shape[:2]
    geojson = export.features_to_geojson(
        features, map_name, out_w, out_h,
        crop_offset=prepared["crop_offset"],
        cropped=prepared["cropped"],
        geo_transform=geo_transform,
        geo_info=geo_info,
        legend_summary=legend_summary,
    )
    export.write_geojson(geojson, output_dir, map_name)
    export.write_shapefile(geojson, output_dir, map_name)
    export.write_legend(legend_entries, legend_summary, output_dir, map_name)

    # Причина в сводке: сначала про привязку (если её нет), иначе — про уверенность.
    summary_reason = reason
    if not geo_info.get("georeferenced") and geo_info.get("reason"):
        summary_reason = reason or f"без привязки: {geo_info['reason']}"

    return {
        "name": map_name,
        "status": "ok",
        "width": width,
        "height": height,
        "num_features": len(features),
        "num_legend": len(legend_entries),
        "confidence": confidence,
        "reason": summary_reason,
        "georeferenced": geo_info.get("georeferenced", False),
        "crs": f"EPSG:{config.TARGET_EPSG}" if geo_info.get("georeferenced") else "pixel",
        "georef_rms_px": round(geo_info["rms_px"], 3) if geo_info.get("georeferenced") else "",
    }


def _assess_confidence(combined_mask, features):
    """
    Простая эвристика уверенности по доле «найденных» пикселей, числу объектов и тому,
    не жмутся ли объекты к краю кадра (признак пятен/рамки, а не геологии).
    Возвращает (confidence, reason): confidence — 'ok' или 'low'.
    """
    h, w = combined_mask.shape[:2]
    total = h * w
    coverage = cv2.countNonZero(combined_mask) / total if total else 0.0
    num_features = len(features)

    if num_features == 0:
        return "low", "объектов не найдено (вероятно калька/серый чертёж)"
    if coverage < config.LOW_CONFIDENCE_COVERAGE:
        return "low", "очень мало цветных пикселей (вероятно слабый/выцветший скан)"

    edge_frac = _edge_feature_fraction(features, w, h)
    if edge_frac >= config.EDGE_NOISE_FRAC:
        return "low", "много объектов у края (вероятно пятна старения/рамка, не геология)"
    return "ok", ""


def _edge_feature_fraction(features, w, h):
    """Доля объектов, чей центр лежит в краевой полосе кадра (EDGE_BAND_FRAC по сторонам)."""
    if not features:
        return 0.0
    band_x = config.EDGE_BAND_FRAC * w
    band_y = config.EDGE_BAND_FRAC * h
    edge = 0
    for f in features:
        pts = f["points"]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        if cx < band_x or cx > w - band_x or cy < band_y or cy > h - band_y:
            edge += 1
    return edge / len(features)
