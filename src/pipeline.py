"""
pipeline.py — обработка ОДНОЙ карты от начала до конца.

Сейчас это «скелет»: только загрузка скана и сохранение исходника в debug.
По мере роста проекта сюда добавятся этапы: предобработка -> выделение ->
очистка -> векторизация -> экспорт. Каждый этап будет сохранять свой debug-кадр.

Функция возвращает небольшой словарь-отчёт по карте (для сводки _summary.csv).
"""

import cv2

from src import cleanup, config, export, extract, io_utils, preprocess, vectorize


def process_map(image_path, input_dir, output_dir, debug_root, profile_name, debug_enabled=True):
    """
    Прогнать одну карту через пайплайн. Возвращает dict с результатом:
      {name, status, ...}
    status:
      - "failed"  : файл не прочитался (битый/не картинка)
      - "ok"      : обработка прошла (пока что просто загрузка)
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

    # --- Этап 3: выделение объектов (HSV + Canny) ---
    extracted = extract.extract(prepared, profile_name, saver)
    # extracted["color_masks"], extracted["canny"], extracted["combined"]

    # --- Этап 4: очистка масок (морфология + фильтр мелочи) ---
    cleaned = cleanup.cleanup(extracted, saver)
    # cleaned["color_masks"], cleaned["combined"], cleaned["canny"]

    # --- Этап 5: векторизация (контуры -> полилинии) ---
    features = vectorize.vectorize(cleaned, prepared, saver)

    # --- Триаж: насколько уверены в результате ---
    confidence, reason = _assess_confidence(cleaned["combined"], len(features))

    # --- Этап 6: экспорт в GeoJSON ---
    # prepared["color"] мог быть ужат — берём его размер, чтобы координаты совпадали с векторами.
    out_h, out_w = prepared["color"].shape[:2]
    geojson = export.features_to_geojson(
        features, map_name, out_w, out_h,
        crop_offset=prepared["crop_offset"],
        cropped=prepared["cropped"],
    )
    export.write_geojson(geojson, output_dir, map_name)

    return {
        "name": map_name,
        "status": "ok",
        "width": width,
        "height": height,
        "num_features": len(features),
        "confidence": confidence,
        "reason": reason,
    }


def _assess_confidence(combined_mask, num_features):
    """
    Простая эвристика уверенности по доле «найденных» пикселей и числу объектов.
    Возвращает (confidence, reason): confidence — 'ok' или 'low'.
    """
    total = combined_mask.shape[0] * combined_mask.shape[1]
    coverage = cv2.countNonZero(combined_mask) / total if total else 0.0

    if num_features == 0:
        return "low", "объектов не найдено (вероятно калька/серый чертёж)"
    if coverage < config.LOW_CONFIDENCE_COVERAGE:
        return "low", "очень мало цветных пикселей (вероятно слабый/выцветший скан)"
    return "ok", ""
