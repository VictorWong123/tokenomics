"""Optional native check of archived proposal counts, with no model inference."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from bench.common import read_jsonl, write_jsonl
from scripts.check_suffix_replay import check


@unittest.skipUnless(
    os.getenv("TOKENOMICS_SUFFIX_REPLAY_BINARY"),
    "requires optional native suffix replay helper",
)
class NativeSuffixReplayTests(unittest.TestCase):
    def test_archived_counts_and_rejected_expected_count(self):
        binary = Path(os.environ["TOKENOMICS_SUFFIX_REPLAY_BINARY"]).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = check(binary, root / "passed")
            self.assertEqual(result["cases"], 24)
            self.assertFalse(result["model_inference_executed"])
            jobs = read_jsonl(root / "passed/jobs.jsonl")
            first = jobs[0]
            first["expected"]["accepted"] += 1
            bad = root / "bad.jsonl"
            write_jsonl(bad, [first])
            failed = subprocess.run(
                [str(binary), str(bad)], capture_output=True, text=True, timeout=10
            )
            self.assertEqual(failed.returncode, 1)
            self.assertIn("differs from frozen Python counts", failed.stderr)
            summary = json.loads((root / "passed/verification.json").read_text())
            self.assertEqual(summary["state"], "native_suffix_validation_replay_passed")


if __name__ == "__main__":
    unittest.main()
