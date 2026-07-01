<div align="center">

# TerraSoviet · Vectorization and georeferencing of geological maps

**An automated pipeline that brings scanned 1970s Soviet geological maps back to life:
it extracts faults and geological boundaries, turns them into vectors, and ties them
to real-world WGS84 coordinates — no neural-network training, just classic computer vision.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-classic%20CV-5C3EE8?logo=opencv&logoColor=white)
![pyproj](https://img.shields.io/badge/datum-EPSG%3A4284%20to%204326-2E7D32)
![Output](https://img.shields.io/badge/output-GeoJSON%20%2B%20Shapefile-F9A825)
![Tests](https://img.shields.io/badge/tests-pytest%20passing-success)

<sub>Hackathon **"TerraSoviet Data Rescue"** · Track 2 · "Map vectorization and referencing"</sub>

</div>

---

## Contents

- [About the project](#about-the-project)
- [Features](#features)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Example result](#example-result)
- [Usage](#usage)
- [What you get on output](#what-you-get-on-output)
- [Georeferencing: Pulkovo 1942 to WGS84](#georeferencing-pulkovo-1942-to-wgs84)
- [Results on a real dataset](#results-on-a-real-dataset)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Limitations](#limitations)
- [Future work](#future-work)

---

## About the project

For decades, valuable Soviet-era geological data has stayed "frozen" inside scanned maps:
faded paper, creases, pencil marks, coordinates in an outdated datum. These maps cannot be
searched, analyzed, or overlaid onto modern GIS.

**This project automates their digitization.** The input is a folder of scans; the output is
machine-readable **vectors (GeoJSON + Shapefile)** tied to **WGS84**. Everything runs in
batch, without manual annotation and without hardcoded paths: the same code runs on a new
dataset with a single command.

The key principle is **robustness and predictability**. Instead of trying to "perfectly
recognize everything", the pipeline confidently processes clear maps and **explicitly flags**
the hard ones (`low_confidence`, `georeferenced=no`) rather than crashing or passing off
garbage as a result.

---

## Features

| Feature | How it is implemented |
|---|---|
| **Colored feature extraction** | HSV threshold: reds into faults, greens and blues into layer boundaries |
| **Dark line extraction** | *Black-hat* morphology plus a "long and thin" filter |
| **Centerline vectors** | Mask skeletonization and polyline tracing (not a contour "loop") |
| **Georeferencing to WGS84** | Map frame and AOI, homography, Pulkovo 1942 to WGS84 via `pyproj` |
| **Legend extraction** | Colored legend swatches (dense fills) into `<map>.legend.json`, linked to vectors by class |
| **SAM mode (opt.)** | `--use-sam` flag: Segment Anything segmentation as an alternative to HSV; soft fallback if not installed |
| **Noise guard** | Legend and margins cut off by the frame, aging edge-stains removed |
| **Honest triage** | Dubious maps are flagged `low_confidence` with a reason |
| **Two output formats** | GeoJSON (always) plus Shapefile (if `pyshp` is installed) |
| **Debug "comic strip"** | Every stage saves an image to `debug/` for visual inspection |
| **Scalability** | About 100 maps in ~1.5 min, no manual intervention and no crashes |

---

## How it works

The pipeline consists of six sequential stages. Each one saves an intermediate frame to
`debug/`, so the whole process can be "flipped through" by eye.

```
   scan.jpg
      |
      v
[ preprocess ]   resize · CLAHE contrast · denoise
      |
      v
[  extract   ]   HSV color (red/green/blue) · black-hat (dark lines) · Canny
      |          plus a map-frame mask (cuts off margins and legend)
      v
[  cleanup   ]   morphology · small-blob filter · shape filter · edge-stain guard
      |
      v
[ vectorize  ]   skeletonization into centerline polylines (Douglas-Peucker)
      |
      v
[  georef    ]   frame and AOI corners into a homography, then pyproj: EPSG:4284 into EPSG:4326
      |
      v
[  export    ]   GeoJSON · Shapefile · _summary.csv
```

### Why HSV for color

In the **HSV** space, color splits into hue (H), saturation (S), and value/brightness (V).
A "red fault line" is a narrow hue **H** range that barely depends on fading (fading lives in
the value **V**). So an HSV threshold (`cv2.inRange`) gives a simple and robust way to
"cut by color" even on yellowed paper. The red hue is "split" across the ends of the color
wheel, so **two** ranges are used for it.

### Why centerlines instead of contours

If you simply trace a colored blob with a contour, the line turns into a closed "loop"
(points go there and back, the length doubles). Instead we **thin the mask down to a
1-pixel skeleton** (`scikit-image`) and trace it as a real polyline: cleaner geometry and a
correct length.

---

## Quick start

> Requires **Python 3.9+** (tested on 3.14). All dependencies install as prebuilt wheels,
> without GDAL and without compilation.

```bash
# 1. Clone
git clone https://github.com/sakhmedi/tracktwo
cd tracktwo

# 2. Virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Dependencies
pip install -r requirements.txt

# 4. Run on the bundled example (with georeferencing to WGS84)
python main.py --input examples/maps --output output --aoi examples/aoi
```

The result appears in `output/`: `example_original.geojson` (WGS84 coordinates), a Shapefile
and `_summary.csv`. Intermediate frames will be in `debug/example_original/`.

For your own dataset, just drop the scans into `input/` and run `python main.py`.

> **Optional (SAM).** The base pipeline is self-contained. If you want to try
> Segment Anything: `pip install -r requirements-sam.txt`, download the `vit_b` checkpoint into
> `models/` and run with `--use-sam`. Without these files the flag simply falls back to the
> HSV path with a clear message — the pipeline does not crash.

---

## Example result

The "Torgai" map (a colored geological map): the original scan and the extracted centerline
vectors overlaid on top of it.

| Original scan | Detected vectors |
|:---:|:---:|
| ![Original](examples/maps/example_original.jpg) | ![Vectors](examples/example_overlay.jpg) |

An intermediate binary mask of the red faults (after cleanup), from which the vectors are built:

<div align="center">

![Fault mask](examples/example_mask_red.png)

</div>

The finished georeferenced output is in [`examples/example_original.geojson`](examples/example_original.geojson)
(WGS84 coordinates), and an illustrative AOI is in [`examples/aoi/`](examples/aoi). The command
from [Quick start](#quick-start) reproduces this result one to one.

> All images and the GeoJSON in `examples/` were generated by the **current** code, so what
> you see when you run it matches what is shown here. The "Torgai" AOI extent is approximate,
> for demonstrating the Pulkovo 1942 to WGS84 transition.

---

## Usage

```bash
python main.py [--input DIR] [--output DIR] [--debug DIR] [--profile NAME] [--aoi PATH] [--no-debug] [--use-sam]
```

| Argument | Default | Purpose |
|----------|:------------:|------------|
| `--input`    | `input`      | folder of scans (search is **recursive**, into subfolders) |
| `--output`   | `output`     | where to write GeoJSON, Shapefile and the summary |
| `--debug`    | `debug`      | where to write intermediate images |
| `--profile`  | `geological` | set of color thresholds: `geological` or `pencil` |
| `--aoi`      | `(none)`     | folder or file with the Area of Interest for georeferencing. Without it, coordinates stay in pixels |
| `--no-debug` | `(none)`     | do not save intermediate images (faster) |
| `--use-sam`  | `(none)`     | opt. Segment Anything segmentation (needs `requirements-sam.txt` + checkpoint; otherwise falls back to HSV) |

```bash
# Example: an arbitrary folder, no code changes
python main.py --input "path/to/scans" --output output --profile geological --aoi aoi
```

> **No hardcoded paths or file names.** All images from the given folder are processed, so the
> pipeline runs on a new dataset without changes.

---

## What you get on output

In the `output/` folder, for each map:

- **`<map>.geojson`** contains a `FeatureCollection` of lines (`LineString`). Each line's
  properties: the source map, the type (`fault` / `fault_uncertain` / `boundary` / `edge`),
  the source color and the length. `metadata` states whether the map is georeferenced, the
  source and target CRS, as well as the referencing RMS error.
- **`<map>.shp`** (plus `.dbf` / `.shx` / `.prj`) contains the same lines as a Shapefile *(if
  `pyshp` is installed)*.

- **`<map>.legend.json`** (if legend swatches were found on the map) contains the recognized
  colored legend swatches: color, class (`fault`/`boundary`), bbox, mean HSV and a link to
  vectors of the same class. A per-color summary is also duplicated in `metadata.legend`
  inside the GeoJSON. OCR of the layer labels (Cyrillic) is intentionally not done — that is
  a Track 1 task.

Plus one shared file:

- **`_summary.csv`** is a consolidated report over all maps: `status`, `confidence`
  (`ok`/`low`), feature count, legend-swatch count, `georeferenced` (yes/no), `crs`,
  referencing RMS and the flag reason. It shows at a glance where the result is reliable and
  where the map turned out to be hard.

---

## Georeferencing: Pulkovo 1942 to WGS84

The core Track 2 task is not just vectors in pixels, but **vectors in real-world coordinates**.

**How the referencing is built (automatically, without manually placing points):**

1. The rectangular **map frame** (neat-line) is detected, and its 4 corners are taken in pixels.
2. The given **Area of Interest** is taken, its 4 corners in ground coordinates.
3. A **homography** pixel to AOI coordinates is built (`cv2.getPerspectiveTransform`).
4. `pyproj` converts the result from **Pulkovo 1942 (EPSG:4284)** to **WGS84 (EPSG:4326)**.

> **The datum trap.** Soviet coordinates are Pulkovo 1942, not WGS84. If you treat them as
> WGS84, you get a **100+ meter** shift. We use the explicit `EPSG:4284` code and `pyproj`,
> so the transition is correct.

**Output coordinates:**

| Mode | Result |
|---|---|
| **with `--aoi`** | WGS84 `[lon, lat]`, `crs: EPSG:4326`, `georeferenced: true`, RMS in metadata |
| **without `--aoi`** (or frame/AOI not detected) | pixels (x right, y down), `georeferenced: false` (fallback mode) |

**The AOI format** is either a folder of sidecars named after the map (`<map>.geojson` /
`.txt`) **or** a single file for the whole dataset:

- **GeoJSON** contains the area polygon (the outer ring or extent is taken); CRS from the `crs` field.
- **TXT** contains either 4 lines of `lon lat`, or a single line `minlon minlat maxlon maxlat`
  (bbox); an optional first line `# epsg=4284` sets the source CRS.

If the AOI is given in WGS84, specify `# epsg=4326` (or `crs` in the GeoJSON), and the datum
transition becomes the identity.

---

## Results on a real dataset

A run on **94 real hackathon maps** (the two folders of the Track 2 and 3 dataset):

| Metric | Value |
|---|---|
| Processed without crashes | **94 / 94** |
| Time | **~1.5 min** (about 1 s per map) |
| Features extracted | **6,913** |
| Flagged `low_confidence` (triage) | 38 |
| Edge garbage removed by the guard | **~1,800 false lines** |

On **clean colored** geological maps the pipeline confidently traces red faults and green
boundaries as tidy centerlines. **Faded sepia scans** with aging stains are either cleaned up
by the guard or flagged `low_confidence`, so it is immediately clear which maps can be trusted.

---

## Project structure

```
tracktwo/
├── main.py                 # entry point: argument parsing, folder walk, progress
├── requirements.txt        # dependencies (install as wheels, no GDAL)
├── requirements-sam.txt    # OPT. dependencies for --use-sam (torch + segment-anything)
├── src/
│   ├── config.py           # ALL the "magic numbers": HSV/dark-line thresholds, kernels, EPSG
│   ├── io_utils.py         # image discovery, reading (Cyrillic-safe), debug saver
│   ├── preprocess.py       # resize, CLAHE contrast, denoise
│   ├── crop.py             # map-frame (neat-line) detection
│   ├── extract.py          # feature extraction: HSV, black-hat, Canny (+ opt. SAM)
│   ├── cleanup.py          # morphology, shape filters, edge-stain guard
│   ├── vectorize.py        # skeletonization into centerline polylines
│   ├── legend.py           # legend-swatch extraction and linking to vectors
│   ├── sam_extract.py      # OPT. Segment Anything segmenter (soft fallback to HSV)
│   ├── georef.py           # frame and AOI into a homography, then Pulkovo 1942 into WGS84
│   ├── export.py           # writing GeoJSON / Shapefile / legend.json / _summary.csv
│   └── pipeline.py         # gluing all stages for one map plus triage
├── tests/
│   └── test_smoke.py       # end-to-end smoke test
├── examples/               # example input, output, AOI and debug images
├── input/  output/  debug/ # working folders (contents in .gitignore)
```

> All tunable parameters are collected in one place, in **`src/config.py`**. To adapt the
> pipeline to a different dataset, edit the numbers there rather than hunting through the code.

### Stages and debug frames

For each map, a numbered series of frames is saved in `debug/<map>/`:

| Frame (in order) | Stage |
|------|------|
| `original` | the source scan |
| `resized` | shrunk to the working size |
| `clahe` | enhanced contrast (CLAHE on brightness, colors preserved) |
| `denoised` | grayscale plus smoothing |
| `mask_red/green/blue` | "raw" color masks (HSV threshold) |
| `mask_dark` | dark ink lines (black-hat) |
| `canny` | edges (for maps without color) |
| `frame_mask` | the area inside the map frame (margins and legend cut off) |
| `mask_combined` | the merged mask |
| `clean_*` | masks after cleanup (morphology plus shape filter) |
| `clean_combined` | the final clean mask |
| `vectors_overlay` | **centerline vectors over the original**, the main quality check |
| `legend_swatches` | detected legend swatches, boxed (if a legend is present) |

---

## Tests

```bash
python -m pytest
```

The end-to-end smoke test (`tests/test_smoke.py`) runs the pipeline on
`examples/maps/example_original.jpg`, checks GeoJSON validity, the presence of a summary, and
correct fallback when the input folder is missing.

---

## Limitations

- **Color thresholds are not universal.** The HSV ranges are tuned for the scans at hand; on a
  different dataset they may need adjustment. All thresholds live in `src/config.py`, there are
  profiles (`--profile`), and the pipeline **does not crash** on "unclear" maps.
- **Dark lines on maps with relief hatching.** Black-hat catches strong ink lineaments, but
  weak faults are interwoven with the relief hatching and are caught incompletely. The shape
  filter (`DARK_MIN_LENGTH` / `DARK_MAX_THICKNESS`) keeps only long thin components.
- **Tracing paper and gray drawings** are handled weakly (there is no color) and flagged `low`.
- **Georeferencing is approximate.** It is built from the frame and AOI corners (a 4-point
  homography). If the frame is detected imprecisely or the AOI is given as a coarse bbox, a
  shift is possible. The RMS is written to the metadata; without a frame or AOI the map stays
  in pixels rather than being referenced at random.
- **Legend: we cut it off, we don't interpret it.** The legend and margins are removed by the
  frame mask (`MASK_OUTSIDE_FRAME`) so that colored legend swatches do not produce false
  vectors. This is a deliberate choice in favor of a clean result: recognizing the legend
  itself (color swatch to layer label) is not implemented yet. If the frame is not found,
  legend swatches may leak into the result and are caught by triage.
- **Geological cross-sections** are currently not handled separately: the pipeline is tuned for
  plan-view maps. A cross-section will go through the same rules (color/dark lines) but without
  special logic for depth axes and scale.
- **Aging edge-stains** (sepia along the edges) are falsely caught by the red threshold. The
  `DROP_BORDER_TOUCHING` guard removes components touching the frame edge, and the
  `EDGE_NOISE_FRAC` triage flags such maps `low_confidence`.

> This is a **deliberate strategy**: it is better to reliably process a clear subtype of maps
> and explicitly flag the rest than to break on all of them.

---

## Future work

- Georeferencing from the **map's own coordinate grid** (OCR of graticule labels) instead of
  relying on the AOI, more accurate on maps with a labeled frame.
- Stitching distant segments of the same line (currently only nearby ones are stitched).
- Splitting intersecting faults at skeleton junctions.
- **OCR of legend labels** (Cyrillic) on top of the already extracted color swatches — a link
  with Track 1, so that each class gets a human-readable layer name.
- Deepening the **SAM mode** (`--use-sam` already exists): fine-tuning / point prompts instead
  of automatic segmentation, combined with a color prior.

---

<div align="center">
<sub>Made for the <b>TerraSoviet Data Rescue</b> hackathon · Track 2 · classic CV, no neural-network training</sub>
</div>
