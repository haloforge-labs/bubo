"""Called by make-icons.mjs; python -I keeps local modules out of imports."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from PIL import Image, ImageDraw, ImageFont

BUILD = Path(__file__).resolve().parent
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--evidence", type=Path)
args = parser.parse_args()
master = Image.open(BUILD / "appicon.png")
master.load()
assert master.size == (1024, 1024) and master.mode == "RGBA"
assert master.getpixel((0, 0))[3] == 0, "Rounded corners must stay transparent"
ico_path = BUILD / "windows/icon.ico"
master.save(ico_path, format="ICO", sizes=[(s, s) for s in SIZES], bitmap_format="png")
with Image.open(ico_path) as icon:
    assert icon.ico.sizes() == {(s, s) for s in SIZES}
    for size in SIZES:
        frame = icon.ico.getimage((size, size))
        frame.load()
        assert frame.size == (size, size) and frame.mode == "RGBA"
        # Lanczos may leave a few alpha levels of ringing at a transparent corner.
        assert frame.getpixel((0, 0))[3] <= 3

# Read only the standard ICO directory to verify Pillow's 256 px PNG payload.
data = ico_path.read_bytes()
reserved, kind, count = struct.unpack_from("<HHH", data)
assert (reserved, kind, count) == (0, 1, len(SIZES))
frames = []
for i in range(count):
    w, h, _, _, planes, bits, length, offset = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
    png = data[offset:offset + 8] == b"\x89PNG\r\n\x1a\n"
    assert png and offset + length <= len(data)
    frames.append({"size": w or 256, "bytes": length, "png": png, "bits": bits})
assert 1000 < len(data) < 1024 * 1024
summary = {"frames": frames, "files": {}}
for path in (BUILD / "appicon.png", ico_path, BUILD.parent / "frontend/public/favicon.svg"):
    summary["files"][path.name] = {
        "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
    }

if args.evidence:
    # Each row shows 16 / 32 / 48 px at actual size, then 6x nearest-neighbour detail.
    backgrounds = ("#ffffff", "#e6e6e6", "#404040", "#000000")
    font = ImageFont.load_default(size=15)
    preview = Image.new("RGB", (1280, 680), "#b8b8b8")
    draw = ImageDraw.Draw(preview)
    draw.text((12, 10), "A (top) / B (bottom): native 16, 32, 48 px; enlarged 6x below", fill="black", font=font)
    for row, name in enumerate(("A", "B")):
        with Image.open(args.evidence / f"candidate-{name}.png") as candidate:
            for column, background in enumerate(backgrounds):
                x, y = column * 320, 40 + row * 320
                draw.rectangle((x, y, x + 319, y + 319), fill=background)
                ink = "#ffffff" if column > 1 else "#111111"
                draw.text((x + 8, y + 8), f"{name} / {background}", fill=ink, font=font)
                for size, dx in ((16, 30), (32, 130), (48, 240)):
                    frame = candidate.resize((size, size), Image.Resampling.LANCZOS)
                    preview.paste(frame, (x + dx - size // 2, y + 45), frame)
                    draw.text((x + dx - 12, y + 100), str(size), fill=ink, font=font)
                # Enlargements of 16 and 32 px expose antialiasing without smoothing.
                for size, dx in ((16, 12), (32, 116)):
                    frame = candidate.resize((size, size), Image.Resampling.LANCZOS)
                    enlarged = frame.resize((size * 6, size * 6), Image.Resampling.NEAREST)
                    preview.paste(enlarged, (x + dx, y + 125), enlarged)
    preview.save(args.evidence / "preview-comparison.png")
    for row, name in enumerate(("A", "B")):
        preview.crop((0, 40 + row * 320, 1280, 360 + row * 320)).save(args.evidence / f"preview-{name}.png")
    (args.evidence / "icon-validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
