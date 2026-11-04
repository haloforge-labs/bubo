import os
from pathlib import Path
import subprocess
import re
import tempfile
import unittest

JOB = Path(__file__).resolve().parent
# The dry run never opens these paths; only their leaf names are checked.
SCRATCH = Path(tempfile.gettempdir()) / 'bubo-stageb-dryrun'
DEFAULTS = {
    'Runtime': SCRATCH / 'runtimes' / 'kcut-stageb-dryrun',
    'PreviousRuntime': SCRATCH / 'runtimes' / 'kcut-99a643ba',
    'Model': SCRATCH / 'models' / 'Qwen3.8-Flash-Next-Hybrid-GSQ-RCO-Q4KXL-00001-of-00004.gguf',
    'Mtp': SCRATCH / 'models' / 'mtp.gguf',
    'Twin': SCRATCH / 'models' / 'twin.gguf',
    'Manifest': SCRATCH / 'quality-c4-manifest.json',
    'Prompt32K': SCRATCH / 'cold32k.txt',
    'OutRoot': SCRATCH / 'out',
    'TempDir': SCRATCH / 'tmp',
}


class OperatorTests(unittest.TestCase):
    def invoke(self, extra=None):
        env = dict(os.environ, TEMP=str(DEFAULTS['TempDir']), TMP=str(DEFAULTS['TempDir']),
                   GUFO_PREFILL_STAGEB='bad-inherited-value', GUFO_PREFILL_STAGEB_ROWS='0')
        params = {**DEFAULTS, **(extra or {})}
        args = [item for key, value in params.items() for item in ('-' + key, str(value))]
        return subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                               '-File', str(JOB/'gpu-checks-stageb.ps1'), '-DryRun', *args],
                              capture_output=True, text=True, env=env)

    def test_complete_dry_run_and_exact_gates(self):
        p = self.invoke()
        self.assertEqual(p.returncode, 0, p.stderr)
        for step in range(1, 6): self.assertIn(f'STEP {step} DRY-RUN', p.stdout)
        self.assertEqual(p.stdout.count("'--prompt-file'"), 3)
        self.assertEqual(p.stdout.count("'--compare'"), 2)
        self.assertEqual(p.stdout.count("'--per-token-nll'"), 1)
        self.assertIn("'--score-prefill' '0'", p.stdout)
        self.assertIn('5504', p.stdout)
        self.assertIn('GUFO_PREFILL_STAGEB*', p.stdout)
        self.assertIn('stageb_ops_test.exe', p.stdout)
        self.assertIn('--allow-final-trim', p.stdout)
        self.assertIn('--map-out', p.stdout)
        self.assertIn('17.461', p.stdout)
        self.assertIn('1247.9', p.stdout)
        self.assertIn('no executable launched', p.stdout)
        budgets = [float(x) for x in re.findall(r'; budget ([0-9.]+) min:', p.stdout)]
        self.assertEqual(sum(budgets), 28.)
        self.assertIn('one hard group allowance of 28 minutes', p.stdout)
        self.assertIn('output-consistency gate, not an independent validation of which tensor or token was read', p.stdout)

    def test_wrong_model_rejected(self):
        p = self.invoke({'Model': SCRATCH / 'models' / 'wrong.gguf'})
        self.assertNotEqual(p.returncode, 0)


if __name__ == '__main__': unittest.main()
