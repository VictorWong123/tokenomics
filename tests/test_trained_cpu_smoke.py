"""Exercise bounded smoke orchestration using two real synthetic CPU profiles."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.common import ROOT, sha256, write_jsonl
from scripts.trained_cpu_smoke import MODEL_PINS, smoke


class TrainedModelIdentityTests(unittest.TestCase):
    def test_second_model_is_pinned_and_wrong_weights_fail_closed(self):
        self.assertEqual(len(MODEL_PINS["qwen25-15"]["sha256"]), 64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "wrong.gguf"
            model.write_bytes(b"not the pinned 1.5B weights")
            args = SimpleNamespace(
                reference=model,
                candidate=model,
                model=model,
                model_id="qwen25-15",
                out=root / "failed",
            )
            with self.assertRaisesRegex(ValueError, "pinned qwen25-15"):
                smoke(args)
            manifest = json.loads((args.out / "manifest.json").read_text())
            self.assertFalse(manifest["synthetic_fixture"])
            self.assertEqual(manifest["model_id"], "qwen25-15")
            self.assertEqual(manifest["model_pin"], MODEL_PINS["qwen25-15"])
            self.assertFalse((args.out / "complete.json").exists())


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
                candidate_profile=os.getenv(
                    "TOKENOMICS_TEST_CANDIDATE_PROFILE", "greedy"
                ),
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
