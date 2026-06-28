"""
cleanup.py — этап 4: очистка бинарных масок.

После HSV маска «грязная»: одиночные крапинки от бумаги, обрывки букв,
разорванные линии. Здесь мы её причёсываем тремя приёмами:

  1) OPEN  (эрозия+дилатация) — убирает одиночные белые точки-шум.
  2) CLOSE (дилатация+эрозия) — заполняет мелкие дырки и соединяет разрывы линии.
  3) Фильтр по площади — выкидывает «кляксы» меньше N пикселей (буквы, крапинки),
     оставляя крупные вытянутые объекты (линии разломов).

На выходе — те же маски, но чистые. Каждая сохраняется в debug.
"""

import cv2
import numpy as np

from src import config


def cleanup(extracted, saver):
    """
    extracted — словарь из extract: {"color_masks", "canny", "combined"}.
    Возвращает структуру с очищенными масками:
      {"color_masks": {name: {"mask": clean, "type": ...}}, "combined": clean, "canny": clean|None}
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE),
    )

    clean_color_masks = {}
    combined = None

    # Чистим каждую цветную маску по отдельности.
    for color_name, spec in extracted["color_masks"].items():
        # Для разломов (линий) включаем «мост», чтобы сшить разорванные штрихи.
        is_fault = spec["type"].startswith("fault")
        # Тёмные линии (fault_uncertain) шумные: дополнительно оставляем только
        # длинные тонкие компоненты, выкидывая буквы/штриховку/заливки.
        line_only = spec["type"] == "fault_uncertain"
        clean = _clean_mask(spec["mask"], kernel, bridge=is_fault, line_only=line_only)
        clean_color_masks[color_name] = {"mask": clean, "type": spec["type"]}
        saver.save(f"clean_{color_name}", clean)

        # Собираем общую чистую маску заново из очищенных кусков.
        if combined is None:
            combined = clean.copy()
        else:
            combined = cv2.bitwise_or(combined, clean)

    # Если цветных масок не было (профиль pencil) — берём края Canny как основу.
    clean_canny = None
    if extracted["canny"] is not None:
        # У краёв не выкидываем мелкое так агрессивно — только соединяем разрывы.
        clean_canny = cv2.morphologyEx(extracted["canny"], cv2.MORPH_CLOSE, kernel,
                                       iterations=config.MORPH_CLOSE_ITERATIONS)
        saver.save("clean_canny", clean_canny)

    if combined is None:
        # Нет цвета вообще — общей маской становится очищенный Canny (или пусто).
        h, w = extracted["combined"].shape[:2]
        combined = clean_canny if clean_canny is not None else np.zeros((h, w), dtype=np.uint8)

    saver.save("clean_combined", combined)

    return {"color_masks": clean_color_masks, "combined": combined, "canny": clean_canny}


def _clean_mask(mask, kernel, bridge=False, line_only=False):
    """
    Применить OPEN -> CLOSE -> (опц. МОСТ) -> фильтр мелких компонентов к одной маске.
    bridge=True добавляет ещё один CLOSE с бОльшим ядром, чтобы сшить разорванные линии.
    line_only=True дополнительно оставляет только длинные тонкие компоненты (для
    тёмных линий: отсекает буквы, штриховку рельефа и заливки).
    """
    # Гард от краевых пятен — РАНО, до морфологии: пока пятно старения ещё одно
    # связное пятно, касающееся края, оно убирается целиком. Если ждать до фильтра
    # площадей, OPEN раздробит пятно на куски, и часть «отлипнет» от края, уцелев.
    if config.DROP_BORDER_TOUCHING:
        mask = _drop_border_components(mask)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                              iterations=config.MORPH_OPEN_ITERATIONS)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel,
                              iterations=config.MORPH_CLOSE_ITERATIONS)

    if bridge and config.BRIDGE_KERNEL_SIZE:
        bridge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.BRIDGE_KERNEL_SIZE, config.BRIDGE_KERNEL_SIZE),
        )
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, bridge_kernel,
                                  iterations=config.BRIDGE_ITERATIONS)

    if line_only:
        return _keep_line_like(closed)
    return _remove_small_components(closed, config.MIN_COMPONENT_AREA)


def _border_tolerance(mask):
    """Сколько пикселей от края считаем «касанием» (доля от длинной стороны)."""
    h, w = mask.shape[:2]
    return max(1, int(round(config.BORDER_TOUCH_TOLERANCE_FRAC * max(h, w))))


def _apply_label_keep(labels, keep):
    """Собрать маску из меток, которые надо оставить (векторно, без цикла по пикселям)."""
    keep[0] = False  # метка 0 — фон, никогда не оставляем
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _border_touch_flags(stats, w_img, h_img, tol):
    """Булев вектор по меткам: True там, где bbox компоненты касается края кадра."""
    x = stats[:, cv2.CC_STAT_LEFT]
    y = stats[:, cv2.CC_STAT_TOP]
    w = stats[:, cv2.CC_STAT_WIDTH]
    h = stats[:, cv2.CC_STAT_HEIGHT]
    return ((x <= tol) | (y <= tol)
            | ((x + w) >= (w_img - tol)) | ((y + h) >= (h_img - tol)))


def _drop_border_components(mask):
    """
    Убрать ВСЕ связные компоненты, касающиеся края кадра (независимо от размера).
    Применяется к «сырой» маске: цельное краевое пятно/рамка/поля уходят одним куском,
    а внутренние объекты (разломы, границы) остаются нетронутыми.
    """
    h_img, w_img = mask.shape[:2]
    tol = _border_tolerance(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = ~_border_touch_flags(stats, w_img, h_img, tol)
    return _apply_label_keep(labels, keep)


def _keep_line_like(mask):
    """
    Оставить только ДЛИННЫЕ и ТОНКИЕ связные компоненты (линии), выкинув короткие
    кляксы и толстые пятна. Критерий: длинная сторона bbox >= DARK_MIN_LENGTH И
    средняя толщина (площадь / длинная сторона) <= DARK_MAX_THICKNESS.
    """
    h_img, w_img = mask.shape[:2]
    tol = _border_tolerance(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    w = stats[:, cv2.CC_STAT_WIDTH]
    h = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    long_side = np.maximum(w, h)
    thickness = np.divide(area, long_side, out=np.zeros(len(area), dtype=float),
                          where=long_side > 0)
    keep = (long_side >= config.DARK_MIN_LENGTH) & (thickness <= config.DARK_MAX_THICKNESS)
    if config.DROP_BORDER_TOUCHING:
        keep &= ~_border_touch_flags(stats, w_img, h_img, tol)
    return _apply_label_keep(labels, keep)


def _remove_small_components(mask, min_area):
    """
    Убрать связные белые области площадью меньше min_area пикселей (крапинки, буквы),
    а также краевые компоненты (пятна старения/рамка). Длинные линии внутри остаются.
    """
    h_img, w_img = mask.shape[:2]
    tol = _border_tolerance(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    if config.DROP_BORDER_TOUCHING:
        keep &= ~_border_touch_flags(stats, w_img, h_img, tol)
    return _apply_label_keep(labels, keep)
