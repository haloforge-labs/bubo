"""Make one labelled PNG contact sheet per page from headless scenario captures."""

import argparse
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shots", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-height", type=int, default=4000)
    args = parser.parse_args()
    root = args.shots.resolve(strict=True)
    output = args.output.resolve() if args.output else root / "contact-sheets"
    output.mkdir(exist_ok=True)
    font_path = (
        Path(__file__).resolve().parents[1]
        / "wails/frontend/src/fonts/Go-Mono.ttf"
    )
    font = ImageFont.truetype(str(font_path), 12)
    title_font = ImageFont.truetype(str(font_path), 20)
    columns, width, height, header = 4, 340, 276, 56
    rows_per_part = (args.max_height - header) // height
    if rows_per_part < 1:
        parser.error("--max-height must fit at least one row and its header")
    sheets = screenshots = 0
    variants = {"light-100": 0, "dark-100": 1, "light-150": 2, "dark-150": 3}

    def order(path: Path) -> tuple[str, int]:
        case, theme, scale = path.stem.rsplit("-", 2)
        return case, variants[f"{theme}-{scale}"]

    for page in sorted(root.iterdir()):
        if not page.is_dir() or page == output:
            continue
        images = sorted(page.glob("*.png"), key=order)
        if not images:
            continue
        count_per_part = rows_per_part * columns
        parts = math.ceil(len(images) / count_per_part)
        for part in range(parts):
            batch = images[part * count_per_part : (part + 1) * count_per_part]
            sheet = Image.new(
                "RGB",
                (columns * width, header + math.ceil(len(batch) / columns) * height),
                "#e8e5df",
            )
            draw = ImageDraw.Draw(sheet)
            draw.text(
                (12, 12),
                f"{page.name} — {len(images)} captures · part {part + 1}/{parts}",
                font=title_font,
                fill="#171717",
            )
            for index, path in enumerate(batch):
                x, y = (index % columns) * width, header + (index // columns) * height
                with Image.open(path) as original:
                    if original.format != "PNG":
                        raise ValueError(f"Expected PNG: {path}")
                    thumb = ImageOps.contain(original.convert("RGB"), (width - 16, 224))
                    sheet.paste(thumb, (x + (width - thumb.width) // 2, y))
                label = "\n".join(textwrap.wrap(path.stem, width=43))
                draw.multiline_text(
                    (x + 8, y + 230), label, font=font, fill="#171717", spacing=2
                )
            suffix = f"-{part + 1:02}" if parts > 1 else ""
            target = output / f"{page.name}{suffix}.png"
            sheet.save(target, format="PNG")
            sheets += 1
            print(f"{target}: {len(batch)} screenshots, {sheet.width} x {sheet.height}")
        screenshots += len(images)
    if not sheets:
        raise SystemExit("No page screenshots found")
    print(f"contact sheets: {sheets} passed, 0 failed; {screenshots} input PNGs")


if __name__ == "__main__":
    main()
