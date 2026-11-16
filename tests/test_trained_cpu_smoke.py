"""Exercise bounded smoke orchestration using two real synthetic CPU profiles."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.common import ROOT, sha256, write_jsonl
from scripts.trained_cpu_smoke import smoke


@unittest.skipUnless(
    os.getenv("TOKENOMICS_TEST_BINARY") and os.getenv("TOKENOMICS_GREEDY_BINARY"),
    "requires guarded and greedy-only CPU executables",
)
class TrainedCPUSmokeTests(unittest.TestCase):
    def test_fixture_success_and_wrong_profile_failure_remain_distinct(self):
        from make_fixtures import create

        reference = Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        candidate = Path(os.environ["TOKENOMICS_GREEDY_BINARY"]).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bench").mkdir()
            ids = [
                "development-code-03",
                "development-qa-01",
                "development-code-01",
                "development-summary-01",
            ]
            write_jsonl(
                root / "bench/prompts-v1.jsonl",
                [{"id": pid, "prompt": "abcdefghi" * 2} for pid in ids],
            )
            model = root / "fixture.gguf"
            create(model, 100, ngram_cycle=True)
            args = SimpleNamespace(
                reference=reference,
                candidate=candidate,
                model=model,
                out=root / "passed",
            )
            with patch("scripts.trained_cpu_smoke.ROOT", root):
                result = smoke(args, expected_sha=sha256(model))
                self.assertEqual(result["state"], "passed")
                self.assertEqual(result["checked_comparisons"], 160)
                self.assertTrue(result["synthetic_fixture"])
                self.assertTrue((args.out / "complete.json").is_file())
                manifest = json.loads((args.out / "manifest.json").read_text())
                self.assertNotEqual(
                    manifest["binary_sha256"]["reference"],
                    manifest["binary_sha256"]["greedy"],
                )
                args.out = root / "failed"
                args.candidate = reference
                with self.assertRaisesRegex(ValueError, "Wrong CPU/sampler"):
                    smoke(args, expected_sha=sha256(model))
                status = json.loads((args.out / "status.json").read_text())
                self.assertEqual(status["state"], "failed")
                self.assertFalse((args.out / "complete.json").exists())


if __name__ == "__main__":
    unittest.main()
