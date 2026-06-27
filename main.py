"""
main.py — точка входа. Запускается из командной строки:

    python main.py --input input --output output --debug debug --profile geological

Что делает:
  1) находит все картинки в папке --input (включая подпапки),
  2) для каждой запускает пайплайн (src/pipeline.py),
  3) печатает прогресс и краткий итог.

Пути по умолчанию — относительные папки проекта, поэтому судья может просто
склонировать репозиторий, положить сканы в input/ и запустить `python main.py`.
"""

import argparse
import sys

from src import config, export, io_utils, pipeline


def _force_utf8_output():
    """
    Заставить консоль печатать в UTF-8, иначе на Windows русский текст превращается
    в «кракозябры». reconfigure есть в Python 3.7+; на старых версиях просто пропускаем.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Векторизация исторических геологических карт (Трек 2)."
    )
    parser.add_argument("--input", default="input",
                        help="папка со сканами карт (по умолчанию: input)")
    parser.add_argument("--output", default="output",
                        help="папка для результатов GeoJSON (по умолчанию: output)")
    parser.add_argument("--debug", default="debug",
                        help="папка для промежуточных картинок (по умолчанию: debug)")
    parser.add_argument("--profile", default=config.DEFAULT_PROFILE,
                        choices=list(config.PROFILES.keys()),
                        help="набор цветовых порогов (по умолчанию: %(default)s)")
    parser.add_argument("--aoi", default=None,
                        help="папка или файл с Area of Interest для геопривязки "
                             "(пиксели -> WGS84). Если не задан — координаты пиксельные")
    parser.add_argument("--no-debug", action="store_true",
                        help="не сохранять промежуточные картинки (быстрее)")
    return parser.parse_args()


def main():
    _force_utf8_output()
    args = parse_args()
    debug_enabled = not args.no_debug

    # Создаём папки результата заранее (если их нет).
    io_utils.ensure_dir(args.output)
    if debug_enabled:
        io_utils.ensure_dir(args.debug)

    images = io_utils.find_images(args.input)
    if not images:
        import os
        if not os.path.isdir(args.input):
            print(f"Папка '{args.input}' не найдена. Создайте её и положите туда сканы "
                  f"карт, либо укажите свою папку через --input.")
        else:
            print(f"В папке '{args.input}' не найдено картинок. Поддерживаемые форматы: "
                  f"{', '.join(config.IMAGE_EXTENSIONS)}")
        return

    print(f"Найдено карт: {len(images)}. Профиль: {args.profile}. "
          f"Debug: {'вкл' if debug_enabled else 'выкл'}")
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
        )
        results.append(result)

        status = result["status"]
        mark = "OK " if status == "ok" else "ПРОПУСК"
        print(f"[{i:>3}/{len(images)}] {mark} {result['name']}")

    # Сводный отчёт по всем картам.
    summary_path = export.write_summary(results, args.output)

    # Краткий итог
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = len(results) - ok
    low = sum(1 for r in results if r.get("confidence") == "low")
    geo = sum(1 for r in results if r.get("georeferenced"))
    print("-" * 60)
    print(f"Готово. Успешно: {ok}, пропущено: {failed}, low_confidence: {low}, "
          f"геопривязано (WGS84): {geo}.")
    print(f"GeoJSON -> {args.output}/  |  Сводка -> {summary_path}")


if __name__ == "__main__":
    main()
