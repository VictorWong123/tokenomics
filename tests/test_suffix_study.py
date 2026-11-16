"""Real CPU fixture orchestration; no Metal performance evidence."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.suffix_study import analyze
from bench.common import ROOT, read_jsonl, sha256, write_jsonl
from bench.run import run


@unittest.skipUnless(os.getenv("TOKENOMICS_TEST_BINARY"), "requires fixture runtime")
class SuffixStudyTests(unittest.TestCase):
    def test_complete_grid_and_tampering(self):
        from make_fixtures import create

        binary = Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bench").mkdir()
            (root / "configs").mkdir()
            model = root / "repeat.gguf"
            create(model, 100, ngram_cycle=True)
            config = json.loads((ROOT / "configs/suffix-screen-v1.json").read_text())
            config.update(context=64, threads=1)
            (root / "configs/suffix-screen-v1.json").write_text(json.dumps(config))
            prompts = [
                {
                    "id": pid,
                    "prompt": "abcdefghi" * 2,
                    "category": "synthetic",
                    "split": "development",
                }
                for pid in config["prompt_ids"]
            ]
            write_jsonl(root / "bench/prompts-v1.jsonl", prompts)
            run(
                binary,
                model,
                None,
                prompts,
                config,
                root / "screen",
                identity={"synthetic_fixture": True},
            )
            with patch("analysis.suffix_study.ROOT", root), patch(
                "analysis.suffix_study.MODEL_SHA256", sha256(model)
            ), patch("analysis.suffix_study.metal_profile") as profile:
                result = analyze(root / "screen")
                self.assertEqual(result["records"], 72)
                self.assertEqual(profile.call_count, 72)
                self.assertGreater(result["work_totals"]["suffix1"]["accepted"], 0)
                self.assertIn("suffix1/ngram1", result["comparisons"])
                rows = read_jsonl(root / "screen/requests.jsonl")
                rows[0]["execution"]["accepted"] += 1
                write_jsonl(root / "screen/requests.jsonl", rows)
                with self.assertRaises(ValueError):
                    analyze(root / "screen")


if __name__ == "__main__":
    unittest.main()
