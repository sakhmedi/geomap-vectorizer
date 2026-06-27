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
        is_fault = spec["type"] == "fault"
        clean = _clean_mask(spec["mask"], kernel, bridge=is_fault)
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


def _clean_mask(mask, kernel, bridge=False):
    """
    Применить OPEN -> CLOSE -> (опц. МОСТ) -> фильтр мелких компонентов к одной маске.
    bridge=True добавляет ещё один CLOSE с бОльшим ядром, чтобы сшить разорванные линии.
    """
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

    filtered = _remove_small_components(closed, config.MIN_COMPONENT_AREA)
    return filtered


def _remove_small_components(mask, min_area):
    """
    Убрать связные белые области площадью меньше min_area пикселей.
    Так уходят крапинки и мелкие буквы, а длинные линии остаются.
    """
    # connectedComponentsWithStats нумерует все «острова» белого и даёт их площади.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    # Метка 0 — это фон, его пропускаем (начинаем с 1).
    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == label] = 255
    return out
