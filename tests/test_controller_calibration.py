"""Calibration provenance and frozen-grid execution with real fixture processes."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bench.controller import (
    _calibration_system_info,
    calibrate,
    configurations,
    point_from_frozen,
    validate_binding,
)
from bench.common import atomic_json, sha256
from bench.run import run
from bench.replay import investigate


class BackendIdentity(unittest.TestCase):
    def test_model_calibration_uses_draft_loaded_backend_identity(self):
        configurations = [
            dict(id="target", policy="target"),
            dict(id="fixed1", policy="fixed", k=1, drafter="model"),
            dict(id="fixed2", policy="fixed", k=2, drafter="model"),
        ]
        prefix = "backend; devices=MTL0,CPU,"
        target_load = "; load_tensors: offloaded 25/25 layers to GPU\n"
        draft_load = "; load_tensors: offloaded 29/29 layers to GPU\n"
        suffix = "; profile=guarded"
        target_only = prefix + target_load + suffix
        target_and_draft = prefix + target_load + draft_load + suffix
        rows = [
            dict(configuration_id="target", system_info=target_only),
            dict(configuration_id="fixed1", system_info=target_and_draft),
            dict(configuration_id="fixed2", system_info=target_and_draft),
        ]
        self.assertEqual(
            _calibration_system_info(rows, configurations, "model"),
            target_and_draft,
        )
        for changed_target in (
            target_only.replace("MTL0", "CPU0"),
            target_only.replace("25/25", "24/25"),
        ):
            with self.subTest(changed_target=changed_target):
                rows[0]["system_info"] = changed_target
                with self.assertRaisesRegex(ValueError, "Mixed calibration backends"):
                    _calibration_system_info(rows, configurations, "model")
        rows[0]["system_info"] = target_only
        rows[-1]["system_info"] = target_and_draft.replace("MTL0", "CPU0")
        with self.assertRaisesRegex(ValueError, "Mixed calibration backends"):
            _calibration_system_info(rows, configurations, "model")


@unittest.skipUnless(os.getenv("TOKENOMICS_TEST_BINARY"), "requires fixture engine")
class Calibration(unittest.TestCase):
    def test_freeze_replay_binding_ablations_and_tampering(self):
        exe = Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        model = exe.parent / "fixtures/ngram-cycle.gguf"
        cfg = dict(
            split="development",
            phase="controller_fit",
            output_caps=[64],
            repetitions=1,
            seed=20261108,
            context=256,
            batch=32,
            threads=1,
            warmup_tokens=2,
            warmup_prompt="a b",
            configurations=[{"id": "target", "policy": "target"}]
            + [
                {"id": f"fixed{a}", "policy": "fixed", "k": a, "drafter": "ngram"}
                for a in (1, 2, 4, 6, 8)
            ],
        )
        prompts = [
            dict(
                id="development-qa-01", category="qa", split="development", prompt="a b"
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run(
                exe,
                model,
                None,
                prompts,
                cfg,
                root / "fit",
                identity={"synthetic_fixture": True},
            )
            investigate(
                exe, model, None, root / "fit", root / "replay", cfg, drafter="ngram"
            )
            prior = calibrate(
                root / "fit", root / "replay", "ngram", synthetic_fixture=True
            )
            self.assertTrue(prior["synthetic"])
            self.assertEqual(len(prior["priors"]), 6)
            self.assertTrue(all(p["base_ms"] > 0 for p in prior["priors"]))
            self.assertEqual(prior["sync_ms_per_token"], 0)
            frozen = json.loads((root / "fit/run_manifest.json").read_text())["frozen"]
            point = point_from_frozen(frozen, "ngram")
            validate_binding(prior, point, synthetic_fixture=True)
            mapped = {**frozen, "configuration_binary_sha256": {"alternate": "0" * 64}}
            with self.assertRaisesRegex(ValueError, "operating point"):
                validate_binding(
                    prior,
                    point_from_frozen(mapped, "ngram", configuration_id="alternate"),
                    synthetic_fixture=True,
                )
            with self.assertRaisesRegex(ValueError, "non-synthetic"):
                validate_binding(prior, point)
            with self.assertRaisesRegex(ValueError, "backend"):
                validate_binding(
                    prior, point, synthetic_fixture=True, system_info="different"
                )
            mutated = copy.deepcopy(prior)
            mutated["provenance"]["operating_point"]["binary_sha256"] = "0" * 64
            with patch("bench.run.subprocess.run") as spawn, patch(
                "bench.run.source_fingerprint", return_value={}
            ):
                with self.assertRaisesRegex(ValueError, "operating point"):
                    run(
                        exe,
                        model,
                        None,
                        prompts,
                        {**cfg, "configurations": configurations(mutated)},
                        root / "bad",
                        identity={"synthetic_fixture": True},
                    )
                spawn.assert_not_called()
            validation = [{**prompts[0], "id": "development-qa-03"}]
            measured = run(
                exe,
                model,
                None,
                validation,
                {
                    **cfg,
                    "output_caps": [32],
                    "selected_fixed": prior["provenance"]["selected_fixed"],
                    "configurations": configurations(prior),
                },
                root / "validation",
                identity={"synthetic_fixture": True},
            )
            from analysis.controller import report

            summary = report(root / "validation")
            self.assertTrue(summary["synthetic_fixture"])
            self.assertFalse(summary["heldout"])
            self.assertIn("cost-aware/target", summary["comparisons"])
            self.assertLess(
                summary["stage_costs"]["cost-aware"]["hot_policy_fraction"], 1
            )
            self.assertTrue(
                (root / "validation/analysis/controller-comparison.png").exists()
            )
            self.assertEqual(len(measured), 7)
            self.assertTrue(all(r["correctness"] == "pass" for r in measured))
            for cid in ("cost-aware", "no-fallback", "no-probes"):
                row = next(r for r in measured if r["configuration_id"] == cid)
                self.assertTrue(row["controller_active"])
                self.assertEqual(
                    row["settings"]["controller"]["prior_id"], prior["prior_id"]
                )
            requests = next((root / "replay").glob("*/checkpoint-*/requests.jsonl"))
            requests.write_text(requests.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                calibrate(
                    root / "fit", root / "replay", "ngram", synthetic_fixture=True
                )

    def test_model_calibration_measures_catchup_and_requires_it(self):
        exe = Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        model = exe.parent / "fixtures/ngram-cycle.gguf"
        cfg = dict(
            split="development",
            phase="controller_fit",
            output_caps=[64],
            repetitions=1,
            seed=20261108,
            context=256,
            batch=32,
            threads=1,
            warmup_tokens=2,
            warmup_prompt="a b",
            configurations=[dict(id="target", policy="target")]
            + [
                dict(id=f"fixed{a}", policy="fixed", k=a, drafter="model")
                for a in (1, 2, 4, 6, 8)
            ],
        )
        prompts = [
            dict(
                id="development-qa-01", category="qa", split="development", prompt="a b"
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run(
                exe,
                model,
                model,
                prompts,
                cfg,
                out / "fit",
                identity={"synthetic_fixture": True},
            )
            investigate(
                exe, model, model, out / "fit", out / "replay", cfg, drafter="model"
            )
            prior = calibrate(
                out / "fit", out / "replay", "model", synthetic_fixture=True
            )
            self.assertGreater(prior["sync_ms_per_token"], 0)
            self.assertEqual(
                prior["provenance"]["operating_point"]["draft_sha256"], sha256(model)
            )
            import subprocess, sys

            destination = out / "exported.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bench.controller",
                    "--fit-dir",
                    str(out / "fit"),
                    "--replay-dir",
                    str(out / "replay"),
                    "--drafter",
                    "model",
                    "--out",
                    str(destination),
                    "--synthetic-fixture",
                ],
                check=True,
                capture_output=True,
            )
            self.assertEqual(json.loads(destination.read_text()), prior)
            transition = next((out / "replay").glob("*/transition/complete.json"))
            transition.unlink()
            with self.assertRaisesRegex(ValueError, "actual catch-up"):
                calibrate(out / "fit", out / "replay", "model", synthetic_fixture=True)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.getenv("TOKENOMICS_TEST_BINARY"), "requires fixture engine")
class StudyPipeline(unittest.TestCase):
    def test_complete_study_and_resume_are_frozen(self):
        from scripts.controller_study import execute_study, correctness_gate

        exe = Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        model = exe.parent / "fixtures/ngram-cycle.gguf"
        cfg = json.loads(
            (
                Path(__file__).resolve().parents[1] / "configs/controller-study-v1.json"
            ).read_text()
        )
        cfg.update(
            fit_prompt_ids=["development-qa-01"],
            validation_prompt_ids=["development-qa-03", "development-code-04"],
            output_caps=[16, 32],
            repetitions=1,
            context=256,
            batch=32,
            threads=1,
            warmup_tokens=2,
        )
        prompts = [
            dict(id=pid, category=pid.split("-")[1], split="development", prompt="a b")
            for pid in cfg["fit_prompt_ids"] + cfg["validation_prompt_ids"]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            gate = correctness_gate(
                exe, model, model, prompts, cfg, out / "gate", "model"
            )
            self.assertEqual(gate["state"], "passed")
            rows, result, logging = execute_study(
                exe, model, None, prompts, cfg, out / "study", synthetic_fixture=True
            )
            self.assertEqual(len(rows), 28)
            self.assertIn("cost-aware/acceptance", result["comparisons"])
            self.assertIn("cost-aware-detailed/cost-aware", logging["comparisons"])
            digest = sha256(out / "study/prior.json")
            resumed, _, _ = execute_study(
                exe, model, None, prompts, cfg, out / "study", synthetic_fixture=True
            )
            self.assertEqual(rows, resumed)
            self.assertEqual(digest, sha256(out / "study/prior.json"))
            import shutil
            from scripts.verify_controller_evidence import check
            from bench.common import source_fingerprint

            shutil.copytree(out / "gate", out / "study/gate-model")
            correctness_gate(
                exe, model, None, prompts, cfg, out / "study/gate-ngram", "ngram"
            )
            atomic_json(
                out / "study/controller-build.json",
                dict(
                    binary_sha256=sha256(exe),
                    configuration=dict(
                        values=dict(
                            TOKENOMICS_CPU_BATCH_INVARIANT="ON",
                            TOKENOMICS_CPU_PACKED_VERIFY="OFF",
                        )
                    ),
                ),
            )
            atomic_json(
                out / "study/status.json",
                dict(
                    state="controller_study_complete",
                    peak_rss_bytes=max(r["peak_rss_bytes"] for r in rows),
                ),
            )
            receipt = check(
                out / "study",
                source=source_fingerprint()["revision"],
                synthetic_fixture=True,
            )
            self.assertEqual(receipt["verified_validation_records"], 28)
            from scripts.verify_controller_evidence import check_partial
            from analysis.controller import report

            partial = out / "partial"
            shutil.copytree(out / "study", partial)
            shutil.rmtree(partial / "logging")
            (partial / "validation/complete.json").unlink()
            blocks = sorted((partial / "validation/blocks").iterdir())
            for block in blocks[1:]:
                shutil.rmtree(block)
            for name in ("requests.jsonl", "correctness.jsonl"):
                records = [
                    json.loads(line)
                    for line in (partial / "validation" / name).read_text().splitlines()
                ]
                records = [r for r in records if r["block_id"] == blocks[0].name]
                (partial / "validation" / name).write_text(
                    "".join(json.dumps(r) + "\n" for r in records)
                )
            verified_partial = check_partial(
                partial, source=source_fingerprint()["revision"], synthetic_fixture=True
            )
            self.assertTrue(verified_partial["partial"])
            self.assertEqual(
                verified_partial["verified_records_by_stage"]["validation"], 7
            )
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                report(partial / "validation")
            from analysis.controller_details import generate

            details = generate(
                out / "study/validation", fit=out / "study/fit", out=out / "details"
            )
            self.assertTrue(details["synthetic_fixture"])
            self.assertIn("qa", details["categories"])
            self.assertGreater(
                details["policies"]["cost-aware"]["decision_update_ns_per_cycle"], 0
            )
            for name in (
                "memory.png",
                "ablations-and-regressions.png",
                "fit-cost-acceptance.png",
            ):
                self.assertTrue((out / "details" / name).exists())
            from scripts.evaluate_controller import evaluate, prepare

            eval_prompts = [
                {**p, "id": "heldout-" + p["id"], "split": "heldout"}
                for p in prompts[:2]
            ]
            prior = json.loads((out / "study/prior.json").read_text())
            template = {**cfg, "output_caps": [16], "repetitions": 1}
            with self.assertRaisesRegex(ValueError, "correctly labeled"):
                prepare(prior, digest, eval_prompts, template)
            self.assertEqual(
                evaluate(
                    exe,
                    model,
                    None,
                    out / "study/prior.json",
                    eval_prompts,
                    template,
                    out / "heldout",
                    synthetic_fixture=True,
                ),
                0,
            )
            heldout_result = json.loads(
                (out / "heldout/evaluation/analysis/summary.json").read_text()
            )
            self.assertTrue((out / "heldout/prompt-fit/complete.json").exists())
            from bench.prompt_fit import check as check_prompt_fit

            with self.assertRaisesRegex(ValueError, "exceed context"):
                check_prompt_fit(
                    exe,
                    model,
                    eval_prompts,
                    {**template, "output_caps": [512]},
                    out / "oversize",
                )
            self.assertTrue(heldout_result["heldout"])
            self.assertTrue(heldout_result["synthetic_fixture"])
            self.assertEqual(
                heldout_result["selected_fixed"], prior["provenance"]["selected_fixed"]
            )
            from scripts.demo_controller import render

            demonstration = render(out / "study", max_cycles=8)
            self.assertIn("Synthetic transformer fixtures", demonstration)
            self.assertIn(
                json.loads((out / "study/prior.json").read_text())["prior_id"],
                demonstration,
            )
            self.assertIn("exact tokens", demonstration)
            path = out / "study/validation/analysis/summary.json"
            summary = json.loads(path.read_text())
            summary["comparisons"]["cost-aware/target"]["ratio"] += 0.1
            atomic_json(path, summary)
            with self.assertRaisesRegex(ValueError, "regenerated statistics"):
                check(
                    out / "study",
                    source=source_fingerprint()["revision"],
                    synthetic_fixture=True,
                )
