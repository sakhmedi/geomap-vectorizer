"""
sam_extract.py — an OPTIONAL segmenter based on the Segment Anything Model (SAM).

Why: mentor's advice — for complex maps you can use a pretrained SAM instead of the
color threshold. This addresses the "innovation / AI/ML frameworks" criterion. But SAM
pulls in heavy dependencies (torch + segment-anything + a checkpoint ~370 MB for vit_b),
so this path is DISABLED by default and enabled by the --use-sam flag.

The project's key principle is "don't crash": if torch / segment-anything / the checkpoint
are unavailable, we print a clear hint and return None. The calling code (extract.py) then
silently stays on the classic HSV path. This way the default pipeline is always reproducible
out of the box, and SAM is an honest bonus for those who installed it.

Installing SAM:  pip install -r requirements-sam.txt
Checkpoint:      put sam_vit_b_01ec64.pth into models/ (or set SAM_CHECKPOINT).
"""

import os

import cv2
import numpy as np

from src import config, extract

# Print the "SAM missing" hint only once per run, not for every map.
_WARNED = False


def available():
    """
    Check whether SAM can actually be run: are the packages installed and does the
    checkpoint file exist. Does not import anything heavy unless necessary.
    """
    try:
        import torch  # noqa: F401
        import segment_anything  # noqa: F401
    except ImportError:
        return False
    return os.path.isfile(_checkpoint_path())


def _checkpoint_path():
    """The checkpoint path: the SAM_CHECKPOINT environment variable takes precedence over the config."""
    return os.environ.get("SAM_CHECKPOINT", config.SAM_CHECKPOINT)


def _warn_unavailable():
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    print("  [SAM] --use-sam was given, but SAM is unavailable (no torch/segment-anything "
          "or no checkpoint). Staying on the classic HSV path. "
          "Install: pip install -r requirements-sam.txt; checkpoint -> "
          f"{_checkpoint_path()}")


# Load the model once and cache it (re-reading the checkpoint is expensive).
_GENERATOR = None


def _get_generator():
    """Lazily create a SamAutomaticMaskGenerator. None if something went wrong."""
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
    except Exception as exc:  # pragma: no cover - depends on the external environment
        print(f"  [SAM] failed to initialize the model: {exc}")
        return None


def extract_color_masks(color_image, profile, saver=None):
    """
    Segment the map with SAM and assemble masks by the profile's color classes.

    Idea: SAM produces a set of segments without labels. We take the BOUNDARIES of the
    elongated segments (region contours = probable geological boundaries/lineaments) and
    assign each to a class by the region's mean color (via the same profile HSV ranges).
    This way SAM becomes an alternative boundary detector, and the shared cleanup/vectorize
    stages follow.

    Returns a dict {color_name: mask} or None if SAM is unavailable/has no result.
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
        print(f"  [SAM] segmentation error: {exc}")
        return None

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    h_img, w_img = color_image.shape[:2]
    img_area = float(h_img * w_img)

    # Blank boundary masks for each profile color.
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
        # Take the region boundary (contour), not the fill — that is the layer line.
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
    """Assign a region to a profile class by its mean HSV (or None if it fits none)."""
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
    """A thin region boundary = the mask minus its erosion (a contour ~1-2 px thick)."""
    seg255 = (seg > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(seg255, kernel, iterations=1)
    return cv2.subtract(seg255, eroded)
