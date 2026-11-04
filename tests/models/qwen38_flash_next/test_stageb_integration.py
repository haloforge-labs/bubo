"""Private control registry and unchanged model arithmetic contracts."""
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]


class IntegrationTests(unittest.TestCase):
    def test_private_registry(self):
        rows = json.loads((ROOT / 'src/cli/serve/release_switches.json').read_text())['switches']
        by_name = {r['name']: r for r in rows}
        for name in ['GUFO_PREFILL_STAGEB', 'GUFO_PREFILL_STAGEB_LAYERS',
                     'GUFO_PREFILL_STAGEB_STRIDE', 'GUFO_PREFILL_STAGEB_ROWS']:
            row = by_name[name]
            self.assertEqual(row['class'], 'INTERNAL')
            self.assertFalse(row['echo'])
            self.assertFalse(row['gui_control'])
            self.assertIsNone(row['values']['default'])

    def test_original_combine_translation_unit_identical(self):
        relative = 'src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp'
        old = subprocess.check_output(['git', 'show', f'99a643ba:{relative}'], cwd=ROOT)
        self.assertEqual((ROOT / relative).read_bytes().replace(b'\r\n', b'\n'), old)

    def test_diagnostic_output_signature_and_arithmetic_flags(self):
        source = (ROOT / 'src/models/qwen38_flash_next/kernels/rocm/stageb.hpp').read_text()
        self.assertIn('const void* expert_out', source)
        self.assertIn('const float* weights', source)
        self.assertIn('const float* inject', source)
        self.assertIn('StageBTotal* totals, StageBRow* samples', source)
        cmake = (ROOT / 'src/models/qwen38_flash_next/CMakeLists.txt').read_text()
        self.assertIn('-fno-fast-math;-ffp-contract=off', cmake)


if __name__ == '__main__':
    unittest.main()
