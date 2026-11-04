"""CPU-only real process cleanup/deadline tests, with a sleeping fake child."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

JOB = Path(__file__).resolve().parent


class SlotTimeoutTests(unittest.TestCase):
    def invoke(self, mode):
        temp = tempfile.TemporaryDirectory(prefix='stageb-slot-')
        self.addCleanup(temp.cleanup)
        p = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                            '-File', str(JOB / 'slot-timeout-harness.ps1'), mode,
                            '-OutputRoot', temp.name],
                           capture_output=True, text=True, timeout=20)
        self.assertEqual(p.returncode, 0, p.stderr)
        record = json.loads(p.stdout[p.stdout.rfind('\n{'):].strip())
        self.assertEqual(record['status'], 'FAIL', record)
        self.assertIn('group timeout', record['error'].lower())
        self.assertLess(record['seconds'], 2.5)
        root = Path(record['output'])
        process = json.loads((root / 'fake-child.process.json').read_text(encoding='utf-8-sig'))
        self.assertIsNone(process['exit_code'])
        pid_text = (root / 'fake-child.out').read_text(encoding='utf-8-sig').strip() if (root / 'fake-child.out').exists() else ''
        if mode == 'prelaunch':
            self.assertFalse(pid_text, 'expired group launched a child')
        else:
            self.assertLess(process['timeout_seconds'], 10)
            self.assertTrue(pid_text, 'fake child must start before cleanup is tested')
            pid = int(pid_text)
            check = subprocess.run(['powershell', '-NoProfile', '-Command',
                                    f'if(Get-Process -Id {pid} -ErrorAction SilentlyContinue){{exit 1}}'],
                                   capture_output=True)
            self.assertEqual(check.returncode, 0, 'sleeping child survived cleanup')
            if mode == 'cumulative':
                first = json.loads((root / 'first.process.json').read_text(encoding='utf-8-sig'))
                self.assertEqual(first['exit_code'], 0)
                self.assertLessEqual(process['timeout_seconds'], 1.5 - first['group_seconds'])

    def test_stalled_child_killed_at_group_deadline(self):
        self.invoke('stall')

    def test_expired_group_rejected_before_launch(self):
        self.invoke('prelaunch')

    def test_second_child_gets_remaining_group_allowance(self):
        self.invoke('cumulative')


if __name__ == '__main__':
    unittest.main()
