"""
extract.py — этап 3: выделение объектов. САМЫЙ важный этап.

Две независимые ветки:
  1) Цвет (HSV): переводим цветной кадр в HSV и для каждого «целевого цвета»
     из профиля (config.PROFILES) вырезаем пиксели в заданном диапазоне -> бинарная маска.
  2) Края (Canny): по серому кадру ищем границы. Это запасной путь для калек,
     где цвета нет.

На выходе — словарь с масками. Белое (255) = «здесь объект», чёрное (0) = фон.
Каждая маска сохраняется в debug, чтобы видеть глазами, что именно поймалось.
"""

import cv2
import numpy as np

from src import config, crop


def extract(prepared, profile_name, saver, use_sam=False):
    """
    prepared — словарь из preprocess: {"color": BGR, "gray": серый}.
    profile_name — какой набор цветов брать из config.PROFILES.
    saver — DebugSaver для промежуточных кадров.
    use_sam — если True и SAM доступен, дополнить цветовые маски сегментами SAM.

    Возвращает:
      {
        "color_masks": {"red": {"mask": ndarray, "type": "fault"}, ...},
        "canny": ndarray | None,
        "combined": ndarray,   # объединение всех цветных масок (для наглядности)
      }
    """
    color_image = prepared["color"]
    gray_image = prepared["gray"]
    profile = config.PROFILES.get(profile_name, {})

    # Переводим в HSV один раз (дальше все цвета режем из него).
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    color_masks = {}
    # Пустая «нулевая» маска нужного размера, в неё накапливаем объединение.
    combined = np.zeros(gray_image.shape, dtype=np.uint8)

    # --- Ветка 1: цвет ---
    for color_name, spec in profile.items():
        mask = _mask_for_color(hsv, spec["ranges"])
        color_masks[color_name] = {"mask": mask, "type": spec["type"]}
        combined = cv2.bitwise_or(combined, mask)
        saver.save(f"mask_{color_name}", mask)

    # --- Ветка 1c (опц.): SAM как альтернативный сегментатор границ слоёв ---
    # Тяжёлый путь, только под флагом. Если SAM/torch/чекпойнт недоступны —
    # sam_extract печатает подсказку и возвращает None (тихий фолбэк на классику).
    if use_sam:
        from src import sam_extract
        sam_masks = sam_extract.extract_color_masks(color_image, profile, saver)
        if sam_masks:
            for color_name, sam_mask in sam_masks.items():
                if color_name in color_masks:
                    # Объединяем с цветовой маской того же класса (SAM дополняет HSV).
                    color_masks[color_name]["mask"] = cv2.bitwise_or(
                        color_masks[color_name]["mask"], sam_mask)
                else:
                    ctype = profile.get(color_name, {}).get("type", "boundary")
                    color_masks[color_name] = {"mask": sam_mask, "type": ctype}
                combined = cv2.bitwise_or(combined, sam_mask)
            saver.save("mask_sam_combined", combined)

    # --- Ветка 1b: тёмные линии (разломы чернилами/карандашом) ---
    # На многих картах разломы — тёмные, а не цветные; HSV их не ловит.
    if config.EXTRACT_DARK_LINES:
        dark = _extract_dark_lines(gray_image)
        color_masks["dark"] = {"mask": dark, "type": "fault_uncertain"}
        combined = cv2.bitwise_or(combined, dark)
        saver.save("mask_dark", dark)

    # --- Ветка 2: края (Canny) ---
    canny = None
    if config.USE_CANNY:
        canny = cv2.Canny(gray_image, config.CANNY_THRESHOLD_LOW, config.CANNY_THRESHOLD_HIGH)
        saver.save("canny", canny)

    # --- Маскируем всё за пределами рамки карты (поля, легенда, штамп) ---
    if config.MASK_OUTSIDE_FRAME:
        frame_mask = _frame_interior_mask(color_image)
        if frame_mask is not None:
            for spec in color_masks.values():
                spec["mask"] = cv2.bitwise_and(spec["mask"], frame_mask)
            combined = cv2.bitwise_and(combined, frame_mask)
            if canny is not None:
                canny = cv2.bitwise_and(canny, frame_mask)
            saver.save("frame_mask", frame_mask)

    saver.save("mask_combined", combined)

    return {"color_masks": color_masks, "canny": canny, "combined": combined}


def _frame_interior_mask(color_image):
    """
    Бинарная маска внутренней области рамки карты (белое = внутри рамки).
    None, если рамка не найдена — тогда ничего не маскируем (безопасный откат).
    """
    quad = crop.find_map_corners(color_image)
    if quad is None:
        return None
    h, w = color_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    return mask


def _extract_dark_lines(gray):
    """
    Выделить тёмные тонкие линии (вероятные разломы) операцией black-hat.

    Black-hat = closing(gray) - gray: подсвечивает тёмные структуры тоньше ядра
    на более светлом фоне. Дальше порог -> бинарная маска тёмных штрихов.
    Форму (длинные/тонкие vs кляксы) отфильтрует этап cleanup.
    """
    k = config.DARK_BLACKHAT_KERNEL
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, config.DARK_THRESHOLD, 255, cv2.THRESH_BINARY)
    return mask


def color_mask(hsv, ranges):
    """Публичная обёртка над _mask_for_color (нужна legend.py для тех же порогов)."""
    return _mask_for_color(hsv, ranges)


def _mask_for_color(hsv, ranges):
    """
    Собрать одну бинарную маску для цвета, у которого может быть НЕСКОЛЬКО диапазонов
    HSV (например, красный — два диапазона по краям круга тонов).
    Маски диапазонов объединяем логическим ИЛИ.
    """
    h, w = hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for lower, upper in ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        part = cv2.inRange(hsv, lower_np, upper_np)
        mask = cv2.bitwise_or(mask, part)
    return mask
