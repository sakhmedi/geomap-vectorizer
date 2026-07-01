"""
config.py — the single place holding all the project's "magic numbers".

Why: when you want to tweak exactly which red counts as a "fault", or how
aggressively to clean noise, you change the number HERE instead of hunting for it
across the whole codebase. This also helps on the judge's hidden dataset: it is
enough to adjust the thresholds in one file.

Important about HSV in OpenCV: the ranges are NOT the textbook ones!
  H (hue):         0..179   (not 0..360)
  S (saturation):  0..255
  V (value):       0..255
"""

# ----------------------------------------------------------------------------
# 1. Preprocessing (stage 2)
# ----------------------------------------------------------------------------

# If the scan is huge (e.g. 6000px on the long side) — shrink it, so processing
# is faster and there is less noise. 0 = do not resize.
MAX_IMAGE_SIDE = 2000  # pixels on the long side

# Crop to the map frame (a stage after resizing). OFF by default: on creased
# historical maps the rectangular frame is detected unreliably (it may cut off a corner).
# This is an optional feature — enabled by setting True. It is safe: if the frame is not
# found, the frame is not cropped. The area fractions bound which contour counts as the
# "map frame".
CROP_TO_MAP_BORDER = False
CROP_MIN_AREA_FRAC = 0.30   # the frame must occupy at least 30% of the image
CROP_MAX_AREA_FRAC = 0.95   # and no more than 95% (otherwise it is the whole sheet, not the frame)

# Mask out everything OUTSIDE the detected map frame (margins, legend, stamp).
# Safe: if the frame is not found — nothing is masked. Unlike cropping, the
# coordinates do not shift (we only zero out the masks outside the frame), so
# georeferencing and debug frames stay consistent. Helps against false positives
# on colored legend swatches and margin notes.
MASK_OUTSIDE_FRAME = True

# CLAHE — "smart" local contrast (helps with fading).
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Denoise strength (the larger, the stronger the smoothing of paper/pencil grain).
DENOISE_STRENGTH = 5  # odd number for the median filter, 0 = disable


# ----------------------------------------------------------------------------
# 2. Color profiles for feature extraction (stage 3) — the HEART of the tuning
# ----------------------------------------------------------------------------
#
# A profile = a dict "color name -> list of HSV ranges".
# Each range is (lower HSV bound, upper HSV bound).
# Red needs TWO ranges, because the red hue is "split" across the ends of the
# H wheel (around 0 and around 179) — so we catch both and merge them.
#
# Each feature's type (fault/boundary) ends up in the GeoJSON properties.

PROFILES = {
    # Profile for colored geological maps.
    "geological": {
        "red": {
            "type": "fault",          # red lines are treated as faults
            "ranges": [
                ((0, 70, 50),   (10, 255, 255)),    # red near the start of the wheel
                ((170, 70, 50), (179, 255, 255)),   # red near the end of the wheel
            ],
        },
        "green": {
            "type": "boundary",       # green lines/dikes — layer boundaries
            # Thresholds from real measurements: olive dikes are dull (H~35-40, S~35+, V~90+),
            # so S/V are lowered. H starts at 33 — above the paper-background hue (H~15),
            # so the background hue does NOT get caught.
            "ranges": [
                ((33, 30, 30), (95, 255, 220)),
            ],
        },
        "blue": {
            "type": "boundary",
            "ranges": [
                ((90, 30, 30), (140, 255, 220)),
            ],
        },
    },

    # Profile for tracing paper / gray drawings: no color, we rely on edges (Canny).
    # An empty color dict = the pipeline goes by edges only.
    "pencil": {},
}

# The default profile if --profile is not given on the command line.
DEFAULT_PROFILE = "geological"


# ----------------------------------------------------------------------------
# 3. Edge detection (stage 3, branch for maps without color)
# ----------------------------------------------------------------------------

# Whether to use Canny in addition to color. For tracing paper it is the only path.
USE_CANNY = True
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150


# ----------------------------------------------------------------------------
# 3b. Dark lines (stage 3) — faults/lineaments drawn in DARK ink
# ----------------------------------------------------------------------------
#
# On many Soviet maps the faults and geological lines are dark strokes, not
# colored ones. HSV does not see them. We extract them with the "black hat"
# operation: it highlights thin dark structures on a light background. Then a
# threshold -> a binary mask. The type of such lines is 'fault_uncertain' (a dark
# line = a probable fault, but without color, so we mark it as less confident).
EXTRACT_DARK_LINES = True
DARK_BLACKHAT_KERNEL = 15        # black-hat kernel size (odd): width of the "catchable" line
DARK_THRESHOLD = 30              # black-hat brightness threshold (0..255): higher = stricter
# Shape filter for dark components: keep only the LONG and THIN ones (lines),
# discard short blobs and thick patches (letters, relief hatching, fills).
DARK_MIN_LENGTH = 60             # min. long side of the component bbox, px
DARK_MAX_THICKNESS = 6.0         # max. mean thickness = area / long_side, px


# ----------------------------------------------------------------------------
# 4. Mask cleanup (stage 4)
# ----------------------------------------------------------------------------

# Morphology "brush" size: the larger, the more aggressively we remove noise/close gaps.
MORPH_KERNEL_SIZE = 3            # pixels (odd)
MORPH_OPEN_ITERATIONS = 1       # remove single noise dots
MORPH_CLOSE_ITERATIONS = 2      # close small line gaps

# Discard "blobs" smaller than this number of pixels (letters, crumbs, spots).
MIN_COMPONENT_AREA = 80

# Guard against edge-stains and the frame: remove connected components TOUCHING the
# image edge. Why: real faults and boundaries lie inside the map frame (neat-line) and
# do not reach the very edge of the sheet. But brownish-orange aging stains, the frame
# shadow/line and the margins do touch the edge — and their red tint is falsely caught
# as a "fault" (especially on sepia scans). This cuts off the main source of false
# positives. If the map frame is found and MASK_OUTSIDE_FRAME is on, this filter simply
# does not trigger anyway (nothing inside the frame touches the edge), so the two
# mechanisms do not conflict but complement each other.
DROP_BORDER_TOUCHING = True
BORDER_TOUCH_TOLERANCE_FRAC = 0.012   # "touch" = a component within 1.2% of the edge

# "Bridge" for fault lines: an extra CLOSE with a larger kernel, to join the
# broken strokes of one line into a solid segment. Applied ONLY to fault-type
# masks, and BEFORE the small-blob filter — so the stitched strokes survive the filter.
BRIDGE_KERNEL_SIZE = 7      # pixels (odd); 0 = disable the bridge
BRIDGE_ITERATIONS = 1


# ----------------------------------------------------------------------------
# 5. Vectorization (stage 5)
# ----------------------------------------------------------------------------

# Douglas-Peucker: how much to simplify the polyline (larger = fewer points, coarser).
APPROX_EPSILON = 2.0            # pixels

# Contours shorter than this (in points) are discarded as garbage.
MIN_CONTOUR_POINTS = 5

# Centerlines via skeletonization: instead of tracing a blob with a contour "loop"
# (points are doubled), we thin the mask down to a 1px line and trace it as a polyline.
# Requires scikit-image; if it is absent — automatic fallback to contours.
USE_SKELETON = True
# Skeleton segments shorter than this (in pixels of total length) are treated as noise.
MIN_SKELETON_LENGTH = 15


# ----------------------------------------------------------------------------
# 6. Triage / confidence (stage 6)
# ----------------------------------------------------------------------------

# If the fraction of colored pixels in the mask is below this fraction of the whole
# image — we consider that color "did not fire" (probably tracing paper) and flag
# low_confidence.
LOW_CONFIDENCE_COVERAGE = 0.001   # 0.1% of the area

# Edge-noise triage: if most of the found features huddle near the image edge — they
# are most likely aging stains/the frame, not geology. We flag the map low_confidence
# even if there are many features (protection against a deceptively large count on
# sepia scans).
EDGE_NOISE_FRAC = 0.5     # fraction of "edge" features above which we lower confidence
EDGE_BAND_FRAC = 0.06     # the "edge band" is 6% of the size along each side


# ----------------------------------------------------------------------------
# 6b. Georeferencing (a stage after vectorization) — Track 2 "map referencing"
# ----------------------------------------------------------------------------
#
# The default source CRS of the AOI coordinates. Mentor's advice: Soviet maps are
# Pulkovo 1942 (EPSG:4284), NOT WGS84. If the AOI is given in WGS84 — specify EPSG:4326
# in the AOI file itself (the crs field / the '# epsg=4326' line), and the transition
# becomes the identity.
DEFAULT_SOURCE_EPSG = 4284   # Pulkovo 1942
TARGET_EPSG = 4326           # WGS84


# ----------------------------------------------------------------------------
# 6c. Legend extraction (Track 2: "legend extraction")
# ----------------------------------------------------------------------------
#
# A map legend is a table "color swatch -> geological layer name".
# We do not just cut the legend off as noise: in a separate pass we find the colored
# SWATCHES in it (dense rectangular fills, not thin lines) and link each swatch to a
# feature type (fault/boundary) and to the already extracted vectors of the same
# color. The result is written to <map>.legend.json and a summary is put into the
# GeoJSON metadata. OCR of the layer labels (Cyrillic) is a Track 1 task; here we
# deliberately skip OCR so as not to pull in Tesseract and to stay reproducible
# out of the box.
EXTRACT_LEGEND = True
# The legend swatch size as a fraction of the image area: not a speck and not half the map.
LEGEND_MIN_SWATCH_AREA_FRAC = 0.00015
LEGEND_MAX_SWATCH_AREA_FRAC = 0.05
# A legend swatch is a DENSE fill: area / bbox_area is high. Fault lines are
# thin and winding, their ratio is low, so they do not end up here.
LEGEND_MIN_FILL_RATIO = 0.55
# And roughly compact (not strongly elongated), to cut off pieces of lines.
LEGEND_ASPECT_MIN = 0.25
LEGEND_ASPECT_MAX = 4.0
# Legend swatches are usually OUTSIDE the map frame (on the margins) — we write the
# outside_frame flag into each swatch. But making this a HARD filter is not possible:
# the frame detector often grabs the outer edge of the sheet entirely, and then the
# legend is formally "inside". So by default we do NOT require "outside the frame" and
# rely on the shape (dense compact fill vs a thin line). Set True if on your dataset
# the frame stably coincides with the neat-line and the legend is always outside.
LEGEND_REQUIRE_OUTSIDE_FRAME = False


# ----------------------------------------------------------------------------
# 6d. SAM mode (optional, under the --use-sam flag) — the "innovation" criterion
# ----------------------------------------------------------------------------
#
# Segment Anything Model (mentor's advice) as an ALTERNATIVE to the color threshold
# for maps with complex layer boundaries. Enabled ONLY by the --use-sam flag and pulls
# in heavy dependencies (torch + segment-anything + a checkpoint) from requirements-sam.txt.
# OFF by default: the default pipeline stays lightweight and reproducible out of the box.
# If SAM/torch/checkpoint are unavailable — we print a hint and silently fall back to the
# classic HSV path (the pipeline does not crash).
USE_SAM = False
SAM_MODEL_TYPE = "vit_b"     # vit_b (light) / vit_l / vit_h (more accurate, heavier)
# Path to the checkpoint. Can be overridden by the SAM_CHECKPOINT environment variable.
SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"
# Take only elongated SAM segments (boundaries/lineaments), cutting off large fills.
SAM_MIN_REGION_AREA = 200
SAM_MAX_REGION_AREA_FRAC = 0.25


# ----------------------------------------------------------------------------
# 7. File extensions we treat as maps
# ----------------------------------------------------------------------------

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


# ----------------------------------------------------------------------------
# 8. Debug frames
# ----------------------------------------------------------------------------

# Debug images are only needed for eyeballing, so we shrink them on the long
# side — to avoid filling the disk with gigabytes of PNG. 0 = do not shrink.
DEBUG_MAX_SIDE = 1500  # pixels
