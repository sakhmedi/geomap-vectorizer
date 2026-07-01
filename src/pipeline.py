"""
pipeline.py — processing ONE map from start to finish.

Stages: load -> preprocess -> feature extraction (HSV/dark lines/Canny) ->
mask cleanup -> vectorization (centerlines) -> georeferencing (pixels -> WGS84) ->
export (GeoJSON + Shapefile). Each stage saves its own debug frame.

The function returns a small report dict for the map (for the _summary.csv summary).
Principle: one bad map does not bring down the whole run — we flag status/confidence/georeferenced.
"""

import cv2

from src import (cleanup, config, export, extract, georef, io_utils, legend,
                 preprocess, vectorize)


def process_map(image_path, input_dir, output_dir, debug_root, profile_name,
                debug_enabled=True, aoi_path=None, use_sam=False):
    """
    Run one map through the pipeline. Returns a dict with the result:
      {name, status, ...}
    status:
      - "failed"  : the file could not be read (corrupt/not an image)
      - "ok"      : the map passed the whole pipeline (vectors + export); quality is
                    reflected by the confidence/georeferenced fields, not by status
    """
    map_name = io_utils.get_map_name(image_path, input_dir)

    # --- Stage 1: load ---
    image = io_utils.load_image(image_path)
    if image is None:
        # Don't crash on one bad map — flag it and move on.
        return {"name": map_name, "status": "failed", "reason": "could not read the file"}

    saver = io_utils.DebugSaver(debug_root, map_name, enabled=debug_enabled)
    saver.save("original", image)

    height, width = image.shape[:2]

    # --- Stage 2: preprocessing ---
    prepared = preprocess.preprocess(image, saver)
    # prepared["color"] -> for HSV, prepared["gray"] -> for edges.

    # --- Stage 3: feature extraction (HSV + Canny, opt. SAM) ---
    extracted = extract.extract(prepared, profile_name, saver, use_sam=use_sam)
    # extracted["color_masks"], extracted["canny"], extracted["combined"]

    # --- Stage 4: mask cleanup (morphology + small-blob filter) ---
    cleaned = cleanup.cleanup(extracted, saver)
    # cleaned["color_masks"], cleaned["combined"], cleaned["canny"]

    # --- Stage 5: vectorization (contours -> polylines) ---
    features = vectorize.vectorize(cleaned, prepared, saver)

    # --- Stage 5a: legend extraction (color swatches -> layer class) ---
    legend_entries, legend_summary = legend.extract_legend(
        prepared["color"], profile_name, features=features, saver=saver)

    # --- Triage: how confident we are in the result ---
    confidence, reason = _assess_confidence(cleaned["combined"], features)

    # --- Stage 5b: georeferencing (pixels -> WGS84), if an AOI is given ---
    # We build the referencing in the same pixel space as the vectors (prepared["color"]).
    geo_transform, geo_info = georef.georeference(prepared["color"], map_name, aoi_path)

    # --- Stage 6: export to GeoJSON (+ opt. Shapefile) ---
    # prepared["color"] may have been shrunk — take its size so coordinates match the vectors.
    out_h, out_w = prepared["color"].shape[:2]
    geojson = export.features_to_geojson(
        features, map_name, out_w, out_h,
        crop_offset=prepared["crop_offset"],
        cropped=prepared["cropped"],
        geo_transform=geo_transform,
        geo_info=geo_info,
        legend_summary=legend_summary,
    )
    export.write_geojson(geojson, output_dir, map_name)
    export.write_shapefile(geojson, output_dir, map_name)
    export.write_legend(legend_entries, legend_summary, output_dir, map_name)

    # The summary reason: first about referencing (if there is none), otherwise about confidence.
    summary_reason = reason
    if not geo_info.get("georeferenced") and geo_info.get("reason"):
        summary_reason = reason or f"not georeferenced: {geo_info['reason']}"

    return {
        "name": map_name,
        "status": "ok",
        "width": width,
        "height": height,
        "num_features": len(features),
        "num_legend": len(legend_entries),
        "confidence": confidence,
        "reason": summary_reason,
        "georeferenced": geo_info.get("georeferenced", False),
        "crs": f"EPSG:{config.TARGET_EPSG}" if geo_info.get("georeferenced") else "pixel",
        "georef_rms_px": round(geo_info["rms_px"], 3) if geo_info.get("georeferenced") else "",
    }


def _assess_confidence(combined_mask, features):
    """
    A simple confidence heuristic based on the fraction of "found" pixels, the number of
    features, and whether the features huddle near the image edge (a sign of stains/frame,
    not geology).
    Returns (confidence, reason): confidence is 'ok' or 'low'.
    """
    h, w = combined_mask.shape[:2]
    total = h * w
    coverage = cv2.countNonZero(combined_mask) / total if total else 0.0
    num_features = len(features)

    if num_features == 0:
        return "low", "no features found (probably tracing paper/a gray drawing)"
    if coverage < config.LOW_CONFIDENCE_COVERAGE:
        return "low", "very few colored pixels (probably a weak/faded scan)"

    edge_frac = _edge_feature_fraction(features, w, h)
    if edge_frac >= config.EDGE_NOISE_FRAC:
        return "low", "many features near the edge (probably aging stains/frame, not geology)"
    return "ok", ""


def _edge_feature_fraction(features, w, h):
    """The fraction of features whose center lies in the image edge band (EDGE_BAND_FRAC per side)."""
    if not features:
        return 0.0
    band_x = config.EDGE_BAND_FRAC * w
    band_y = config.EDGE_BAND_FRAC * h
    edge = 0
    for f in features:
        pts = f["points"]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        if cx < band_x or cx > w - band_x or cy < band_y or cy > h - band_y:
            edge += 1
    return edge / len(features)
