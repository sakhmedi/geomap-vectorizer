"""
sam_extract.py — ОПЦИОНАЛЬНЫЙ сегментатор на Segment Anything Model (SAM).

Зачем: совет ментора — для сложных карт можно взять предобученную SAM вместо
цветового порога. Это критерий «инновационность / фреймворки ИИ/МО». Но SAM тянет
тяжёлые зависимости (torch + segment-anything + чекпойнт ~370 МБ для vit_b),
поэтому путь ВЫКЛЮЧЕН по умолчанию и включается флагом --use-sam.

Ключевой принцип проекта — «не падать»: если torch / segment-anything / чекпойнт
недоступны, мы печатаем понятную подсказку и возвращаем None. Вызывающий код
(extract.py) тогда тихо остаётся на классическом HSV-пути. Так дефолтный конвейер
всегда воспроизводим «из коробки», а SAM — честный бонус для тех, кто его поставил.

Установка SAM:  pip install -r requirements-sam.txt
Чекпойнт:       положите sam_vit_b_01ec64.pth в models/ (или задайте SAM_CHECKPOINT).
"""

import os

import cv2
import numpy as np

from src import config, extract

# Печатаем подсказку об отсутствии SAM только один раз за прогон, не на каждой карте.
_WARNED = False


def available():
    """
    Проверить, можно ли реально запустить SAM: установлены ли пакеты и есть ли
    файл чекпойнта. Ничего не импортирует тяжёлого без необходимости.
    """
    try:
        import torch  # noqa: F401
        import segment_anything  # noqa: F401
    except ImportError:
        return False
    return os.path.isfile(_checkpoint_path())


def _checkpoint_path():
    """Путь к чекпойнту: переменная окружения SAM_CHECKPOINT важнее конфига."""
    return os.environ.get("SAM_CHECKPOINT", config.SAM_CHECKPOINT)


def _warn_unavailable():
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    print("  [SAM] --use-sam задан, но SAM недоступен (нет torch/segment-anything "
          "или чекпойнта). Остаюсь на классическом HSV-пути. "
          "Установка: pip install -r requirements-sam.txt; чекпойнт -> "
          f"{_checkpoint_path()}")


# Модель грузим один раз и кэшируем (повторное чтение чекпойнта дорогое).
_GENERATOR = None


def _get_generator():
    """Лениво создать SamAutomaticMaskGenerator. None, если что-то пошло не так."""
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR
    try:
        import torch
        from segment_anything import (SamAutomaticMaskGenerator,
                                       sam_model_registry)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=_checkpoint_path())
        sam.to(device)
        _GENERATOR = SamAutomaticMaskGenerator(sam)
        return _GENERATOR
    except Exception as exc:  # pragma: no cover - зависит от внешней среды
        print(f"  [SAM] не удалось инициализировать модель: {exc}")
        return None


def extract_color_masks(color_image, profile, saver=None):
    """
    Сегментировать карту через SAM и собрать маски по классам цвета профиля.

    Идея: SAM выдаёт набор сегментов без меток. Берём ГРАНИЦЫ вытянутых сегментов
    (контуры регионов = вероятные геологические границы/линеаменты) и относим каждую
    к классу по среднему цвету региона (через те же HSV-диапазоны профиля). Так SAM
    становится альтернативным детектором границ, а дальше идёт общий cleanup/vectorize.

    Возвращает dict {color_name: mask} или None, если SAM недоступен/без результата.
    """
    if not available():
        _warn_unavailable()
        return None
    generator = _get_generator()
    if generator is None:
        return None

    try:
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        masks = generator.generate(rgb)
    except Exception as exc:  # pragma: no cover
        print(f"  [SAM] ошибка сегментации: {exc}")
        return None

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    h_img, w_img = color_image.shape[:2]
    img_area = float(h_img * w_img)

    # Заготовки масок-границ по каждому цвету профиля.
    out = {name: np.zeros((h_img, w_img), dtype=np.uint8) for name in profile}

    for m in masks:
        seg = m.get("segmentation")
        if seg is None:
            continue
        seg = seg.astype(np.uint8)
        area = int(seg.sum())
        if area < config.SAM_MIN_REGION_AREA:
            continue
        if area > config.SAM_MAX_REGION_AREA_FRAC * img_area:
            continue
        color_name = _classify_region(seg, hsv, profile)
        if color_name is None:
            continue
        # Берём границу региона (контур), а не заливку — это и есть линия слоя.
        boundary = _region_boundary(seg)
        out[color_name] = cv2.bitwise_or(out[color_name], boundary)

    out = {name: mask for name, mask in out.items() if cv2.countNonZero(mask) > 0}
    if not out:
        return None

    if saver is not None:
        for name, mask in out.items():
            saver.save(f"sam_{name}", mask)
    return out


def _classify_region(seg, hsv, profile):
    """Отнести регион к классу профиля по среднему HSV (или None, если ни в один)."""
    sel = seg > 0
    if not np.any(sel):
        return None
    mean = hsv[sel].mean(axis=0)
    pixel = np.uint8([[[int(mean[0]), int(mean[1]), int(mean[2])]]])
    for color_name, spec in profile.items():
        for lower, upper in spec["ranges"]:
            if cv2.inRange(pixel, np.array(lower, np.uint8),
                           np.array(upper, np.uint8))[0, 0]:
                return color_name
    return None


def _region_boundary(seg):
    """Тонкая граница региона = маска - её эрозия (контур толщиной ~1-2 px)."""
    seg255 = (seg > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(seg255, kernel, iterations=1)
    return cv2.subtract(seg255, eroded)
