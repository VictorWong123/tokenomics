"""Optional execution/sampling timing profile on actual transformer fixtures."""

import os
from pathlib import Path
import tempfile
import unittest

from bench.replay import run_jobs
from scripts.diagnose_models import build_configuration


@unittest.skipUnless(os.getenv("TOKENOMICS_TEST_BINARY"), "requires fixture runtime")
class VerifyStageTimingTests(unittest.TestCase):
    def test_completed_execution_and_sampling_are_accounted(self):
        binary = Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        if (
            build_configuration(binary)["values"]["TOKENOMICS_VERIFY_STAGE_TIMING"]
            != "ON"
        ):
            self.skipTest("stage timing profile disabled")
        model = binary.parent / "fixtures/ngram-cycle.gguf"
        jobs = [
            {
                "prompt_id": "repeat",
                "prompt": "a b",
                "configuration_id": cid,
                "policy": policy,
                "drafter": drafter,
                "k": 8,
                "repetition": 0,
                "max_tokens": 40,
                "detailed": detailed,
            }
            for cid, policy, drafter, detailed in (
                ("target", "target", "model", True),
                ("ngram", "fixed", "ngram", True),
                ("suffix", "fixed", "suffix", True),
                ("minimal", "fixed", "suffix", False),
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            rows, cycles = run_jobs(
                binary,
                model,
                None,
                jobs,
                Path(tmp),
                {"context": 64, "batch": 32, "threads": 1},
                0,
            )
        for row in rows:
            self.assertIn("TOKENOMICS_VERIFY_STAGE_TIMING=ON", row["system_info"])
            self.assertEqual(row["token_ids"], rows[0]["token_ids"])
            total = row["execution"]
            self.assertGreater(total["target_compute_ms"], 0)
            self.assertGreater(total["target_sampling_ms"], 0)
            self.assertLessEqual(
                total["target_compute_ms"] + total["target_sampling_ms"],
                total["verify_ms"],
            )
            selected = [
                c for c in cycles if c["configuration_id"] == row["configuration_id"]
            ]
            if row["configuration_id"] == "minimal":
                self.assertFalse(selected)
            else:
                for key in ("target_compute_ms", "target_sampling_ms"):
                    self.assertAlmostEqual(
                        sum(c[key] for c in selected), total[key], places=8
                    )
                for cycle in selected:
                    self.assertLessEqual(
                        cycle["target_compute_ms"] + cycle["target_sampling_ms"],
                        cycle["verify_ms"],
                    )


if __name__ == "__main__":
    unittest.main()
