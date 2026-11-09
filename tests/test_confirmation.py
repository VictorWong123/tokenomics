"""Development-frozen ngram confirmation and complete fixture execution."""

import pathlib
import tempfile
import unittest
import os
import json
from bench.common import atomic_json, sha256


class ConfirmationTests(unittest.TestCase):
    def test_confirmation_never_retunes_on_validation_and_rejects_partial_grid(self):
        from scripts.confirm_ngram import analyze

        cfg = {
            "phase": "targeted_confirmation",
            "split": "development",
            "output_caps": [128, 512],
            "repetitions": 3,
            "configurations": [{"id": c} for c in ("target", "fixed1", "fixed8")],
            "selected_fixed": "fixed1",
            "fit_analysis_sha256": "a" * 64,
        }
        prompts = [
            {
                "id": f"development-{cat}-{i:02d}",
                "split": "development",
                "category": cat,
            }
            for cat in ("qa", "code", "summary")
            for i in (3, 4, 5)
        ]
        rows = [
            {
                "prompt_id": p["id"],
                "category": p["category"],
                "output_cap": cap,
                "repetition": rep,
                "configuration_id": cid,
                "correctness": "pass",
                "generated_tokens": 10,
                "generation_ms": ms,
                "warm_ms": ms,
                "load_ms": 1,
                "peak_rss_bytes": 1024,
            }
            for p in prompts
            for cap in cfg["output_caps"]
            for rep in range(3)
            for cid, ms in (("target", 100), ("fixed1", 110), ("fixed8", 50))
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            atomic_json(
                root / "run_manifest.json",
                {"frozen": {"config": cfg, "prompts": prompts}},
            )
            atomic_json(root / "complete.json", {"records": len(rows)})
            path = root / "requests.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            result = analyze(root)
            self.assertEqual(result["selected_fixed"], "fixed1")
            self.assertFalse(result["chosen_fixed_exceeds_practical_margin"])
            self.assertFalse(result["full_feasibility_cleared"])
            self.assertGreater(result["comparisons"]["fixed8/target"]["ratio"], 1)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analyze(root)

    @unittest.skipUnless(
        os.getenv("TOKENOMICS_TEST_BINARY"), "requires built fixture engine"
    )
    def test_confirmation_fixture_executes_complete_frozen_grid(self):
        from unittest.mock import patch
        from bench.run import run
        from scripts.confirm_ngram import analyze
        from scripts import confirm_ngram
        from bench.common import ROOT

        cfg = json.loads((ROOT / "configs/ngram-confirmation-v1.json").read_text())
        cfg.update(
            output_caps=[16, 32],
            repetitions=2,
            context=128,
            batch=32,
            threads=1,
            warmup_tokens=2,
        )
        prompts = [
            {
                "id": pid,
                "category": pid.split("-")[1],
                "split": "development",
                "prompt": "a b",
            }
            for pid in cfg["prompt_ids"]
        ]
        exe = pathlib.Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            rows = run(
                exe,
                exe.parent / "fixtures/ngram-cycle.gguf",
                None,
                prompts,
                cfg,
                out,
                identity={"synthetic_fixture": True},
            )
            result = analyze(out)
            self.assertEqual(len(rows), 108)
            self.assertTrue(result["synthetic_fixture"])
            self.assertFalse(result["chosen_fixed_exceeds_practical_margin"])
            self.assertFalse(result["full_feasibility_cleared"])
            source = out / "source"
            (source / "bench").mkdir(parents=True)
            (source / "build-cpu").symlink_to(exe.parent, target_is_directory=True)
            (source / "bench/prompts-v1.jsonl").write_text(
                "".join(json.dumps(p) + "\n" for p in prompts)
            )
            atomic_json(
                source / "fit.json",
                {
                    "frozen": {
                        "binary_sha256": sha256(exe),
                        "target_sha256": sha256(
                            exe.parent / "fixtures/ngram-cycle.gguf"
                        ),
                        "draft_sha256": None,
                        "config": cfg,
                        "prompts": [{"id": "development-qa-01"}],
                    }
                },
            )
            atomic_json(
                source / "scores.json",
                {"geomean_ratios_vs_target": {"fixed1": 1.01, "fixed8": 0.8}},
            )
            cfg.update(
                fit_manifest="fit.json",
                fit_manifest_sha256=sha256(source / "fit.json"),
                fit_analysis="scores.json",
                fit_analysis_sha256=sha256(source / "scores.json"),
            )
            atomic_json(source / "configs/ngram-confirmation-v1.json", cfg)
            destination = out / "pipeline"
            with patch.object(confirm_ngram, "ROOT", source), patch.object(
                confirm_ngram,
                "download",
                return_value=exe.parent / "fixtures/ngram-cycle.gguf",
            ), patch.object(
                confirm_ngram, "available_memory", return_value=16 * 2**30
            ):
                self.assertEqual(confirm_ngram.pipeline(destination, out / "cache"), 0)
            self.assertTrue((destination / "environment.json").exists())
            status = json.loads((destination / "status.json").read_text())
            self.assertEqual(status["state"], "confirmation_complete")
            self.assertEqual(status["records"], 108)
