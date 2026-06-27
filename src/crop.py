"""
crop.py — необязательный этап: обрезка до области карты (по рамке neat-line).

Зачем: на сканах вокруг карты есть поля и легенда с цветными образцами.
Образцы в легенде дают ложные срабатывания (красный/зелёный «значок» становится вектором).
Если найти прямоугольную рамку карты и обрезать до неё — поля и легенда уходят.

ВАЖНО — безопасный откат: рамку находим не всегда (выцветшие/без рамки карты).
Если уверенного прямоугольника нет, НИЧЕГО не режем и возвращаем кадр как есть.
Лучше не обрезать, чем случайно отрезать кусок карты.
"""

import cv2

from src import config


def crop_to_map(image, saver):
    """
    Попробовать обрезать image до рамки карты.
    Возвращает (cropped_image, (x0, y0), cropped_flag):
      - cropped_image — обрезанный или исходный кадр,
      - (x0, y0) — смещение левого верхнего угла обрезки (0,0 если не резали),
      - cropped_flag — True, если реально обрезали.
    """
    if not config.CROP_TO_MAP_BORDER:
        return image, (0, 0), False

    rect = _find_map_rectangle(image)
    if rect is None:
        # Рамка не найдена — безопасно не трогаем.
        return image, (0, 0), False

    x, y, w, h = rect
    cropped = image[y:y + h, x:x + w]
    saver.save("cropped", cropped)
    return cropped, (x, y), True


def _find_map_rectangle(image):
    """
    Найти крупный ~прямоугольный контур (рамку карты).
    Возвращает (x, y, w, h) bounding box или None, если уверенной рамки нет.
    """
    h_img, w_img = image.shape[:2]
    img_area = h_img * w_img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # Утолщаем края, чтобы разорванная рамка соединилась в замкнутый контур.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        # Рамка должна занимать заметную, но не всю площадь кадра.
        if area < config.CROP_MIN_AREA_FRAC * img_area:
            continue
        if area > config.CROP_MAX_AREA_FRAC * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Ищем именно четырёхугольник (прямоугольную рамку).
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
            best_area = area
            best_box = cv2.boundingRect(approx)

    return best_box
