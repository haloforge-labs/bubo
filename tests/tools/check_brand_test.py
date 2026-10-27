"""The GUI URL guard is independent of the legacy-name allow-list."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('check_brand', ROOT / 'tools/release/check_brand.py')
brand = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(brand)


class GuiLinksTest(unittest.TestCase):
    def test_rejects_legacy_urls_in_production_ui_even_with_test_exceptions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'gui/wails/frontend/src'
            source.mkdir(parents=True)
            (source / 'links.ts').write_text("const url = 'https://github.com/gufo-org/gufo'", encoding='utf-8')
            (source / 'safe.ts').write_text("const url = 'https://github.com/haloforge-labs/bubo'", encoding='utf-8')
            (source / 'links.test.ts').write_text("const rejected = 'https://github.com/gufo-org/gufo'", encoding='utf-8')
            (source / 'links.shots.tsx').write_text("const rejected = 'https://github.com/gufo-org/gufo'", encoding='utf-8')
            errors = brand.gui_urls(root)
            self.assertEqual(len(errors), 1)
            self.assertIn('src/links.ts:1:', errors[0])
            (source / 'links.ts').write_text("const url = 'https://haloforgelabs.com'", encoding='utf-8')
            self.assertEqual(brand.gui_urls(root), [])


if __name__ == '__main__':
    unittest.main()
