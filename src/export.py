"""
export.py — этап 6: запись результатов на диск.

Что пишем:
  1) GeoJSON на каждую карту — найденные линии как FeatureCollection (LineString).
  2) (опционально) Shapefile — те же линии, если установлен пакет pyshp.
  3) Сводка _summary.csv по всем картам — честный отчёт (что получилось, что low/без привязки).

Координаты:
  - Если карта геопривязана (есть GeoTransform) — координаты в WGS84 [lon, lat], crs=EPSG:4326.
  - Если привязки нет (рамка/AOI не найдены) — честно отдаём ПИКСЕЛИ (x вправо, y вниз),
    crs='pixel-coordinates', georeferenced=false. Лучше пиксели, чем привязка наугад.
"""

import csv
import json
from pathlib import Path

from src import config, io_utils

# pyshp — опционально. Если нет, Shapefile просто не пишем (GeoJSON судьям достаточно).
try:
    import shapefile  # pyshp
    _HAS_PYSHP = True
except ImportError:  # pragma: no cover
    _HAS_PYSHP = False


def _feature_coordinates(feature, geo_transform):
    """Вернуть координаты линии: WGS84 [lon, lat], если есть привязка, иначе пиксели."""
    pts = feature["points"]
    if geo_transform is not None:
        lonlat = geo_transform.to_wgs84(pts)
        return [[float(lon), float(lat)] for (lon, lat) in lonlat]
    return [[float(x), float(y)] for (x, y) in pts]


def features_to_geojson(features, map_name, width, height,
                        crop_offset=(0, 0), cropped=False,
                        geo_transform=None, geo_info=None, legend_summary=None):
    """Собрать GeoJSON FeatureCollection из списка фич векторизации."""
    georeferenced = geo_transform is not None
    geo_info = geo_info or {}

    geojson_features = []
    for f in features:
        geojson_features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": _feature_coordinates(f, geo_transform),
            },
            "properties": {
                "source_map": map_name,
                "type": f["type"],         # fault / boundary / edge
                "color": f["color"],       # red / green / blue / none
                "length_px": round(f["length_px"], 1),
            },
        })

    if georeferenced:
        crs = {"type": "name", "properties": {"name": f"EPSG:{config.TARGET_EPSG}"}}
        note = ("WGS84 lon/lat. Georeferenced via map-frame corners -> AOI, "
                f"datum EPSG:{geo_info.get('source_epsg', config.DEFAULT_SOURCE_EPSG)} "
                f"-> EPSG:{config.TARGET_EPSG} (pyproj).")
    else:
        crs = {"type": "name", "properties": {"name": "pixel-coordinates"}}
        note = "Pixel coordinates (x right, y down). Not georeferenced (no AOI/frame)."

    metadata = {
        "source_map": map_name,
        "image_width_px": width,
        "image_height_px": height,
        "cropped_to_map_border": cropped,
        "crop_offset_xy": [int(crop_offset[0]), int(crop_offset[1])],
        "georeferenced": georeferenced,
        "note": note,
    }
    if georeferenced:
        metadata.update({
            "source_crs": f"EPSG:{geo_info.get('source_epsg', config.DEFAULT_SOURCE_EPSG)}",
            "target_crs": f"EPSG:{config.TARGET_EPSG}",
            "gcp_count": geo_info.get("gcp_count"),
            "georef_rms_px": round(geo_info.get("rms_px", 0.0), 3),
        })
    else:
        metadata["georef_reason"] = geo_info.get("reason", "")

    if legend_summary:
        metadata["legend"] = legend_summary

    return {
        "type": "FeatureCollection",
        "crs": crs,
        "metadata": metadata,
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


def write_legend(entries, summary, output_dir, map_name):
    """
    Записать легенду карты в output/<map_name>.legend.json (если есть образцы).
    Возвращает путь или None. Это отдельный сайдкар, чтобы не раздувать GeoJSON
    геометрии; сводка по цветам дополнительно лежит и в metadata.legend.
    """
    if not entries:
        return None
    io_utils.ensure_dir(output_dir)
    out_path = Path(output_dir) / f"{map_name}.legend.json"
    payload = {
        "source_map": map_name,
        "note": ("Цветные образцы легенды (HSV) и связь с векторами того же класса. "
                 "Без OCR подписей (кириллица) — это задача Трека 1."),
        "summary": summary,
        "swatches": entries,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return out_path


def write_shapefile(geojson, output_dir, map_name):
    """
    (Опционально) Записать Shapefile из того же GeoJSON. Возвращает путь или None,
    если pyshp не установлен или фич нет. Тихо пропускаем — это бонус-формат.
    """
    if not _HAS_PYSHP:
        return None
    features = geojson.get("features", [])
    if not features:
        return None

    io_utils.ensure_dir(output_dir)
    out_base = str(Path(output_dir) / map_name)
    writer = shapefile.Writer(out_base, shapeType=shapefile.POLYLINE)
    writer.field("src_map", "C", size=80)
    writer.field("type", "C", size=20)
    writer.field("color", "C", size=10)
    writer.field("length_px", "N", decimal=1)

    for feat in features:
        coords = feat["geometry"]["coordinates"]
        writer.line([coords])
        props = feat["properties"]
        writer.record(props["source_map"], props["type"],
                      props["color"], props["length_px"])
    writer.close()

    # .prj с WKT нужного CRS, чтобы GIS правильно показал слой.
    _write_prj(out_base, geojson["metadata"].get("georeferenced", False))
    return Path(out_base + ".shp")


def _write_prj(out_base, georeferenced):
    """Записать .prj (WKT). Для привязанных карт — WGS84; иначе пропускаем."""
    if not georeferenced:
        return
    try:
        from pyproj import CRS
        wkt = CRS.from_epsg(config.TARGET_EPSG).to_wkt()
        Path(out_base + ".prj").write_text(wkt, encoding="utf-8")
    except Exception:
        pass


def write_summary(results, output_dir):
    """
    Записать сводный отчёт output/_summary.csv по всем картам.
    results — список словарей-отчётов от pipeline.process_map.
    """
    io_utils.ensure_dir(output_dir)
    out_path = Path(output_dir) / "_summary.csv"
    columns = ["name", "status", "confidence", "num_features", "num_legend",
               "georeferenced", "crs", "georef_rms_px", "reason"]

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({
                "name": r.get("name", ""),
                "status": r.get("status", ""),
                "confidence": r.get("confidence", ""),
                "num_features": r.get("num_features", 0),
                "num_legend": r.get("num_legend", 0),
                "georeferenced": "yes" if r.get("georeferenced") else "no",
                "crs": r.get("crs", ""),
                "georef_rms_px": r.get("georef_rms_px", ""),
                "reason": r.get("reason", ""),
            })
    return out_path
