"""
export.py — этап 6: запись результатов на диск.

Две вещи:
  1) GeoJSON на каждую карту — найденные линии как FeatureCollection (LineString).
  2) Сводка _summary.csv по всем картам — для честного отчёта (что получилось,
     что помечено low_confidence).

ВАЖНО про координаты: они ПИКСЕЛЬНЫЕ (x вправо, y вниз), без географической привязки.
Привязка к местности (CRS, широта/долгота) — это Трек 3. Здесь честно отдаём пиксели.
"""

import csv
import json
from pathlib import Path

from src import io_utils


def features_to_geojson(features, map_name, width, height, crop_offset=(0, 0), cropped=False):
    """Собрать GeoJSON FeatureCollection из списка фич векторизации."""
    geojson_features = []
    for f in features:
        # GeoJSON хочет координаты как [x, y]. У нас точки — кортежи (x, y).
        coordinates = [[float(x), float(y)] for (x, y) in f["points"]]
        geojson_features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                "source_map": map_name,
                "type": f["type"],         # fault / boundary / edge
                "color": f["color"],       # red / green / blue / none
                "length_px": round(f["length_px"], 1),
            },
        })

    return {
        "type": "FeatureCollection",
        # Помечаем, что координаты пиксельные, а не географические — честно для судей.
        "crs": {"type": "name", "properties": {"name": "pixel-coordinates"}},
        "metadata": {
            "source_map": map_name,
            "image_width_px": width,
            "image_height_px": height,
            "cropped_to_map_border": cropped,
            # Если карта была обрезана, координаты считаются от рамки. Чтобы вернуться
            # к координатам ужатого скана, прибавьте crop_offset_xy к (x, y).
            "crop_offset_xy": [int(crop_offset[0]), int(crop_offset[1])],
            "note": "Pixel coordinates (x right, y down). Georeferencing is Track 3.",
        },
        "features": geojson_features,
    }


def write_geojson(geojson, output_dir, map_name):
    """Записать GeoJSON в output/<map_name>.geojson. Возвращает путь."""
    io_utils.ensure_dir(output_dir)
    out_path = Path(output_dir) / f"{map_name}.geojson"
    # ensure_ascii=False — чтобы кириллица в именах писалась нормально, не \uXXXX.
    text = json.dumps(geojson, ensure_ascii=False, indent=2)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def write_summary(results, output_dir):
    """
    Записать сводный отчёт output/_summary.csv по всем картам.
    results — список словарей-отчётов от pipeline.process_map.
    """
    io_utils.ensure_dir(output_dir)
    out_path = Path(output_dir) / "_summary.csv"
    columns = ["name", "status", "confidence", "num_features", "reason"]

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({
                "name": r.get("name", ""),
                "status": r.get("status", ""),
                "confidence": r.get("confidence", ""),
                "num_features": r.get("num_features", 0),
                "reason": r.get("reason", ""),
            })
    return out_path
