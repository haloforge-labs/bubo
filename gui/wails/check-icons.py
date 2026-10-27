"""Read-only checks of the shipped icon assets; run after make-icons.mjs."""

from collections import Counter
from pathlib import Path
import re
import struct
import unittest
import xml.etree.ElementTree as ET

from PIL import Image

BUILD = Path(__file__).resolve().parent / 'build'
WAILS = BUILD.parent
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
TILE = (37, 37, 37, 255)
CORAL = (243, 85, 85, 255)
SVG = '{http://www.w3.org/2000/svg}'


class IconAssets(unittest.TestCase):
    def test_master_uses_sidebar_tile_and_preserves_icon_coral(self):
        css = (WAILS / 'frontend/src/styles/app.css').read_text(encoding='utf-8')
        dark = re.search(r":root\[data-theme='dark'\]\s*\{([^}]+)\}", css)[1]
        self.assertRegex(dark, r'--wash:\s*#252525;')
        with Image.open(BUILD / 'appicon.png') as image:
            self.assertEqual((image.size, image.mode), ((1024, 1024), 'RGBA'))
            self.assertEqual(image.getpixel((32, 512)), TILE)
            self.assertEqual(image.getpixel((0, 0))[3], 0)
            self.assertIn(CORAL, {color for _, color in image.getcolors(1024 * 1024)})

    def test_all_windows_frames_have_the_sidebar_tile(self):
        path = BUILD / 'windows/icon.ico'
        data = path.read_bytes()
        self.assertEqual(struct.unpack_from('<HHH', data), (0, 1, len(SIZES)))
        with Image.open(path) as icon:
            self.assertEqual(icon.ico.sizes(), {(size, size) for size in SIZES})
            for index, size in enumerate(SIZES):
                with self.subTest(size=size):
                    width, height, _, _, _, bits, length, offset = struct.unpack_from(
                        '<BBBBHHII', data, 6 + 16 * index
                    )
                    self.assertEqual((width or 256, height or 256, bits), (size, size, 32))
                    self.assertEqual(data[offset:offset + 8], b'\x89PNG\r\n\x1a\n')
                    self.assertLessEqual(offset + length, len(data))
                    frame = icon.ico.getimage((size, size))
                    self.assertLessEqual(frame.getpixel((0, 0))[3], 3)
                    opaque = Counter({
                        color: count for count, color in frame.getcolors(size * size)
                        if color[3] == 255
                    })
                    self.assertEqual(opaque.most_common(1)[0][0], TILE)

    def test_favicon_has_the_generated_tile_and_original_owl_path(self):
        favicon = ET.parse(WAILS / 'frontend/public/favicon.svg').getroot()
        self.assertEqual(favicon.attrib['viewBox'], '0 0 32 32')
        tile = favicon.find(f'{SVG}rect')
        self.assertIsNotNone(tile)
        self.assertEqual(tile.attrib, {
            'width': '32', 'height': '32', 'rx': '7.04', 'fill': '#252525'
        })
        owl = favicon.find(f'{SVG}svg')
        self.assertIsNotNone(owl)
        for name, value in {'x': '2', 'y': '2', 'width': '28', 'height': '28'}.items():
            self.assertEqual(owl.attrib[name], value)
        source = ET.parse(WAILS / 'icons/owl.svg').getroot().find(f'{SVG}path')
        path = owl.find(f'{SVG}path')
        self.assertEqual(path.attrib['d'], source.attrib['d'])
        self.assertEqual(path.attrib['fill'], '#f35555')
        self.assertEqual(path.attrib['stroke'], '#f35555')
        self.assertEqual(path.attrib['stroke-width'], '.9')
        self.assertEqual(path.attrib['stroke-linejoin'], source.attrib['stroke-linejoin'])


if __name__ == '__main__':
    unittest.main()
