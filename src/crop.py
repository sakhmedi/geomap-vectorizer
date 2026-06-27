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


def find_map_corners(image):
    """
    Найти 4 угла прямоугольной рамки карты (как массив точек Nx2 в пикселях),
    или None, если уверенного четырёхугольника нет.

    Это публичная функция: её использует georef.py для построения GCP, независимо
    от того, включена ли обрезка (CROP_TO_MAP_BORDER). Сама обрезка — отдельное
    решение, а углы рамки полезны для геопривязки в любом случае.
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

    best_quad = None
    best_score = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        # Рамка должна занимать заметную, но не всю площадь кадра.
        if area < config.CROP_MIN_AREA_FRAC * img_area:
            continue
        if area > config.CROP_MAX_AREA_FRAC * img_area:
            continue
        peri = cv2.arcLength(c, True)
        # Мятые исторические карты редко дают чистый 4-угольник с одним epsilon —
        # пробуем несколько уровней упрощения и берём первый сходящийся в выпуклый квад.
        approx = None
        for eps in (0.02, 0.03, 0.05, 0.08):
            cand = cv2.approxPolyDP(c, eps * peri, True)
            if len(cand) == 4 and cv2.isContourConvex(cand):
                approx = cand
                break
        if approx is None:
            continue
        # Среди кандидатов предпочитаем самый «прямоугольный»: площадь контура,
        # делённая на площадь его bounding box, у настоящей рамки близка к 1.
        x, y, w, h = cv2.boundingRect(approx)
        rectangularity = area / float(w * h) if w * h else 0.0
        score = rectangularity * area  # прямоугольный И крупный
        if score > best_score:
            best_score = score
            best_quad = approx.reshape(4, 2)

    return best_quad


def _find_map_rectangle(image):
    """
    Найти крупный ~прямоугольный контур (рамку карты).
    Возвращает (x, y, w, h) bounding box или None, если уверенной рамки нет.
    """
    quad = find_map_corners(image)
    if quad is None:
        return None
    return cv2.boundingRect(quad.astype("int32"))
