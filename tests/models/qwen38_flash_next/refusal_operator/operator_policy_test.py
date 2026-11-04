"""Pure collection/analysis tests; starts no process or network connection."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location("driver",Path(__file__).with_name("refusal-check-lqua26.py"))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

class Policy(unittest.TestCase):
    def test_classifier_and_prefix(self):
        self.assertTrue(d.refused("I cannot assist with that."))
        self.assertFalse(d.refused("Here is the requested answer."))
        self.assertEqual(d.prefix([1,2,3],[1,2,4]),2)
        self.assertEqual(d.prefix(list(range(50)),list(range(50))),40)

    def test_complete_answers_precede_counts(self):
        with tempfile.TemporaryDirectory() as path:
            for arm in d.ARMS:
                rows=[]
                for i in range(50):
                    refusal=i==0 and arm=="ista-off"
                    rows.append(dict(i=i,prompt=f"case {i}",refused=refusal,
                                     answer="FULL-ANSWER-MARKER"+"abc"*100,
                                     token_ids=[1,2,3] if arm!="ista-off" else [4,5,6]))
                (Path(path)/f"{arm}.jsonl").write_text("\n".join(map(json.dumps,rows)),encoding="utf-8")
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream): d.compare(path)
            text=stream.getvalue()
            self.assertLess(text.index("FULL-ANSWER-MARKER"),text.index("refusal 1/50"))
            self.assertIn("abc"*100,(Path(path)/"classification-flips.txt").read_text())
            self.assertIn("first token 50/50",text)
            self.assertFalse((Path(path)/"summary.json").exists())

    def test_stream_can_be_skipped_explicitly(self):
        with tempfile.TemporaryDirectory() as path:
            for arm in d.ARMS[:-1]:
                rows=[dict(i=i,prompt=f"case {i}",refused=False,answer="ok",token_ids=[1]) for i in range(50)]
                (Path(path)/f"{arm}.jsonl").write_text("\n".join(map(json.dumps,rows)),encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()): d.compare(path, skip_stream=True)
            self.assertNotIn("ista-stream-1.0",(Path(path)/"readout.txt").read_text())
            with self.assertRaises(FileNotFoundError): d.compare(path)

    def test_missing_ids_fatal(self):
        args=type("Args",(),dict(port=8098,timeout=1))()
        response=dict(choices=[dict(message=dict(content="nonempty"),finish_reason="length")],usage={})
        with patch.object(d,"post",return_value=response):
            with self.assertRaisesRegex(RuntimeError,"missing generated token IDs"): d.response(args,"test")

if __name__=="__main__": unittest.main()
