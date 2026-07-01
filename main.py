"""
main.py — the entry point. Run from the command line:

    python main.py --input input --output output --debug debug --profile geological

What it does:
  1) finds all images in the --input folder (including subfolders),
  2) runs the pipeline (src/pipeline.py) on each one,
  3) prints progress and a short summary.

The default paths are relative project folders, so the judge can simply clone the
repository, drop the scans into input/ and run `python main.py`.
"""

import argparse
import sys

from src import config, export, io_utils, pipeline


def _force_utf8_output():
    """
    Force the console to print in UTF-8, otherwise on Windows non-ASCII text turns into
    "mojibake". reconfigure exists in Python 3.7+; on older versions we simply skip it.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vectorization of historical geological maps (Track 2)."
    )
    parser.add_argument("--input", default="input",
                        help="folder of map scans (default: input)")
    parser.add_argument("--output", default="output",
                        help="folder for GeoJSON results (default: output)")
    parser.add_argument("--debug", default="debug",
                        help="folder for intermediate images (default: debug)")
    parser.add_argument("--profile", default=config.DEFAULT_PROFILE,
                        choices=list(config.PROFILES.keys()),
                        help="set of color thresholds (default: %(default)s)")
    parser.add_argument("--aoi", default=None,
                        help="folder or file with the Area of Interest for georeferencing "
                             "(pixels -> WGS84). If not given — coordinates stay in pixels")
    parser.add_argument("--no-debug", action="store_true",
                        help="do not save intermediate images (faster)")
    parser.add_argument("--use-sam", action="store_true", default=config.USE_SAM,
                        help="opt.: augment feature extraction with SAM segments "
                             "(requires requirements-sam.txt and a checkpoint; otherwise falls back to HSV)")
    return parser.parse_args()


def main():
    _force_utf8_output()
    args = parse_args()
    debug_enabled = not args.no_debug

    # Create the result folders in advance (if they don't exist).
    io_utils.ensure_dir(args.output)
    if debug_enabled:
        io_utils.ensure_dir(args.debug)

    images = io_utils.find_images(args.input)
    if not images:
        import os
        if not os.path.isdir(args.input):
            print(f"Folder '{args.input}' not found. Create it and put map scans there, "
                  f"or specify your own folder via --input.")
        else:
            print(f"No images found in folder '{args.input}'. Supported formats: "
                  f"{', '.join(config.IMAGE_EXTENSIONS)}")
        return

    print(f"Maps found: {len(images)}. Profile: {args.profile}. "
          f"Debug: {'on' if debug_enabled else 'off'}. "
          f"SAM: {'on' if args.use_sam else 'off'}")
    print("-" * 60)

    results = []
    for i, image_path in enumerate(images, start=1):
        result = pipeline.process_map(
            image_path=image_path,
            input_dir=args.input,
            output_dir=args.output,
            debug_root=args.debug,
            profile_name=args.profile,
            debug_enabled=debug_enabled,
            aoi_path=args.aoi,
            use_sam=args.use_sam,
        )
        results.append(result)

        status = result["status"]
        mark = "OK " if status == "ok" else "SKIP"
        print(f"[{i:>3}/{len(images)}] {mark} {result['name']}")

    # The consolidated report over all maps.
    summary_path = export.write_summary(results, args.output)

    # Short summary
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = len(results) - ok
    low = sum(1 for r in results if r.get("confidence") == "low")
    geo = sum(1 for r in results if r.get("georeferenced"))
    legend_total = sum(r.get("num_legend", 0) for r in results)
    print("-" * 60)
    print(f"Done. Successful: {ok}, skipped: {failed}, low_confidence: {low}, "
          f"georeferenced (WGS84): {geo}, legend swatches: {legend_total}.")
    print(f"GeoJSON -> {args.output}/  |  Summary -> {summary_path}")


if __name__ == "__main__":
    main()
