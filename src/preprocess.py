"""
preprocess.py — этап 2: подготовка скана перед выделением объектов.

Карты выцветшие и шумные, поэтому перед анализом их надо «причесать».
На выходе отдаём ДВА варианта одной карты:
  - color : цветной (ужатый + усиленный контраст) — пойдёт в HSV (этап 3),
  - gray  : серый (сглаженный) — пойдёт в поиск краёв Canny (этап 3).

Каждый шаг сохраняет debug-кадр, чтобы эффект было видно глазами.
"""

import cv2

from src import config, crop


def preprocess(image, saver):
    """
    Принять исходный цветной кадр (BGR), вернуть словарь:
        {"color": <BGR для HSV>, "gray": <серый для краёв>}
    saver — DebugSaver, сюда падают промежуточные кадры 01..03.
    """
    # --- Шаг 2.1: ужать большие сканы ---
    # Меньше пикселей = быстрее обработка и меньше мелкого шума.
    resized = _resize_max_side(image, config.MAX_IMAGE_SIDE)
    saver.save("resized", resized)

    # --- Шаг 2.1b: обрезать до рамки карты (безопасно, если найдена) ---
    cropped_img, crop_offset, cropped = crop.crop_to_map(resized, saver)

    # --- Шаг 2.2: усилить контраст (CLAHE) ---
    # Выцветшие цвета становятся ярче/насыщеннее, HSV-порогам легче их поймать.
    # CLAHE применяем к яркости (канал L в пространстве LAB), чтобы НЕ испортить
    # сами цвета (тон остаётся, меняется только контраст по яркости).
    enhanced = _apply_clahe_color(cropped_img)
    saver.save("clahe", enhanced)

    # --- Шаг 2.3: серый + денойз ---
    # Серый нужен для поиска краёв. Денойз убирает зерно бумаги и слабый карандаш,
    # чтобы Canny не принимал их за «края».
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    denoised = _denoise_gray(gray, config.DENOISE_STRENGTH)
    saver.save("denoised", denoised)

    return {
        "color": enhanced,
        "gray": denoised,
        "crop_offset": crop_offset,   # (x0, y0) смещение обрезки в координатах ресайза
        "cropped": cropped,           # True, если реально обрезали
    }


def _resize_max_side(image, max_side):
    """Ужать так, чтобы длинная сторона была не больше max_side. 0 = не трогать."""
    if not max_side:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _apply_clahe_color(image):
    """Усилить контраст по яркости, сохранив цвета (через пространство LAB)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _denoise_gray(gray, strength):
    """Сгладить серый кадр медианным фильтром. strength=0 -> не трогать."""
    if not strength or strength < 3:
        return gray
    ksize = strength if strength % 2 == 1 else strength + 1  # ядро должно быть нечётным
    return cv2.medianBlur(gray, ksize)
