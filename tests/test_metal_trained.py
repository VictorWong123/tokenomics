"""Opt-in trained-model regression for Metal speculative verification."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from bench.common import ROOT, read_jsonl, sha256
from bench.run import run


QWEN_05_SHA256 = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"


@unittest.skipUnless(
    os.getenv("TOKENOMICS_TEST_BINARY") and os.getenv("TOKENOMICS_TEST_METAL_MODEL"),
    "requires the test binary and pinned Qwen2.5 0.5B Metal model",
)
class MetalTrainedRegression(unittest.TestCase):
    def test_fixed4_matches_ordinary_on_original_divergence_prompt(self):
        binary = Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        model = Path(os.environ["TOKENOMICS_TEST_METAL_MODEL"]).resolve()
        self.assertEqual(sha256(model), QWEN_05_SHA256)
        prompt = next(
            row
            for row in read_jsonl(ROOT / "bench/prompts-v1.jsonl")
            if row["id"] == "development-code-03"
        )
        config = json.loads((ROOT / "configs/controller-study-v1.json").read_text())
        config.update(
            phase="metal_trained_regression",
            output_caps=[128],
            repetitions=1,
            context=2048,
            batch=512,
            threads=2,
            detailed=False,
            configurations=[
                {"id": "target", "policy": "target"},
                {
                    "id": "fixed4",
                    "policy": "fixed",
                    "drafter": "ngram",
                    "k": 4,
                },
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            rows = run(
                binary,
                model,
                None,
                [prompt],
                config,
                Path(tmp),
                gpu_layers=99,
                identity={"trained_metal_regression": True},
            )
        self.assertEqual({row["generated_tokens"] for row in rows}, {128})
        self.assertGreater(
            next(row for row in rows if row["configuration_id"] == "fixed4")[
                "execution"
            ]["drafted"],
            0,
        )
        for row in rows:
            self.assertIn("devices=MTL0", row["system_info"])
            self.assertIn("offloaded 25/25 layers to GPU", row["system_info"])
            self.assertIn("TOKENOMICS_METAL_VERIFY_SINGLE_ROW=ON", row["system_info"])


if __name__ == "__main__":
    unittest.main()
