"""
io_utils.py — everything related to files and folders (input/output/debug).

There are deliberately NO computer-vision algorithms here — only "logistics":
find images, create folders, read a scan, save a debug frame.
This way each processing stage can just call save_debug(...) and not think about paths.
"""

from pathlib import Path

import cv2
import numpy as np

from src import config


def find_images(input_dir):
    """
    Return a sorted list of paths to all images in input_dir,
    INCLUDING nested subfolders (rglob = recursive search).

    Why recursive: for the judge (and for us) scans may live not directly in input/
    but in subfolders. No hardcoded names: we take ALL files with a matching extension.
    Sorting — so that the processing order is stable and predictable.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        # Don't crash with a traceback: the judge runs "from scratch", the folder may be absent.
        # Return empty — main.py will print a friendly hint.
        return []

    images = [
        p for p in sorted(input_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    ]
    return images


def load_image(image_path):
    """
    Read an image from disk in color (BGR — OpenCV's channel order).

    Returns a numpy array or None if the file is corrupt/unreadable.
    Important: the file name may contain Cyrillic/special characters — a plain
    cv2.imread on Windows sometimes fails on such paths, so we read via a numpy buffer.
    """
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image  # None if OpenCV could not decode it
    except Exception:
        return None


def ensure_dir(path):
    """Create the folder (and parents) if it does not exist yet. Silently, without errors."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_map_name(image_path, input_dir):
    """
    A unique map name for output files and the debug folder.

    We build it from the RELATIVE path inside input_dir, replacing separators with '__'.
    Example: input='.../track', file='.../track/1/5.jpg'  ->  '1__5'.
    This way files from different subfolders with the same name (1/5.jpg and 2/5.jpg)
    do not overwrite each other.
    """
    rel = Path(image_path).relative_to(Path(input_dir))
    rel_no_ext = rel.with_suffix("")          # drop the .jpg
    return "__".join(rel_no_ext.parts)        # path parts joined by '__'


class DebugSaver:
    """
    A small helper that saves numbered stage images to
    debug/<map_name>/NN_label.png.

    Idea: each stage calls saver.save("clahe", img), and the helper handles the
    numbering (00, 01, 02...) itself. You get a processing "comic strip" to flip through.
    If debug is off (--no-debug), all calls simply do nothing.
    """

    def __init__(self, debug_root, map_name, enabled=True):
        self.enabled = enabled
        self.counter = 0
        self.map_dir = Path(debug_root) / map_name
        if self.enabled:
            # Clear the map folder of frames from the previous run, otherwise old numbers
            # mix with new ones and become confusing.
            if self.map_dir.exists():
                for old in self.map_dir.glob("*.png"):
                    old.unlink()
            ensure_dir(self.map_dir)

    def save(self, label, image):
        """Save a stage frame. label — a short description, e.g. 'clahe' or 'mask_red'."""
        if not self.enabled or image is None:
            return
        filename = f"{self.counter:02d}_{label}.png"
        out_path = self.map_dir / filename
        image = _downscale_for_view(image, config.DEBUG_MAX_SIDE)
        # imencode + tofile — safe writing on Windows with Cyrillic in the path.
        ok, buf = cv2.imencode(".png", image)
        if ok:
            buf.tofile(str(out_path))
        self.counter += 1


def _downscale_for_view(image, max_side):
    """Shrink the image so its long side is no more than max_side. 0 = leave as is."""
    if not max_side:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
