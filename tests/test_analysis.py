"""Frozen analysis grids, uncertainty and development/evaluation separation."""

import unittest
import collections
import pathlib
import tempfile
import json
from bench.common import atomic_json
from analysis.report import cells, paired_summary, summarize, validate_run
from bench.replay import fit_validate


class AnalysisTests(unittest.TestCase):
    def test_screen_cannot_trust_a_forged_passing_correctness_flag(self):
        from analysis.timing_screen import report

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cfg = {
                "split": "development",
                "repetitions": 1,
                "output_caps": [1],
                "configurations": [{"id": c} for c in ("fixed4", "target")],
            }
            atomic_json(
                root / "run_manifest.json",
                {
                    "frozen": {
                        "config": cfg,
                        "prompts": [{"id": "development-qa-00"}],
                        "code": {},
                    }
                },
            )
            atomic_json(root / "complete.json", {"records": 2})
            rows = [
                {
                    "prompt_id": "development-qa-00",
                    "output_cap": 1,
                    "repetition": 0,
                    "configuration_id": cid,
                    "correctness": "pass",
                    "token_ids": [token],
                    "stop": "budget",
                }
                for cid, token in (("fixed4", 2), ("target", 1))
            ]
            (root / "requests.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            (root / "correctness.jsonl").write_text('{"ok": true}\n' * 2)
            with self.assertRaisesRegex(ValueError, "output divergence"):
                report(root, root / "analysis")
            self.assertFalse((root / "analysis").exists())

    def test_frozen_analysis_rejects_missing_or_duplicate_cells_and_retuning(self):
        rows = [
            {
                "prompt_id": "heldout-qa-00",
                "output_cap": 128,
                "configuration_id": c,
                "repetition": rep,
                "correctness": "pass",
                "generated_tokens": 10,
                "generation_ms": duration,
                "category": "qa",
                "warm_ms": duration,
                "load_ms": 1,
                "peak_rss_bytes": 1024,
            }
            for c, duration in (("target", 100), ("fixed2", 200), ("fixed8", 50))
            for rep in range(2)
        ]
        cfg = {
            "split": "heldout",
            "output_caps": [128],
            "repetitions": 2,
            "configurations": [{"id": c} for c in ("target", "fixed2", "fixed8")],
            "selected_fixed": "fixed2",
            "development_selection_sha256": "a" * 64,
        }
        manifest = {
            "frozen": {
                "config": cfg,
                "prompts": [{"id": "heldout-qa-00", "split": "heldout"}],
            }
        }
        fixed = validate_run(rows, manifest, {"records": 6})
        self.assertEqual(fixed, "fixed2")
        self.assertEqual(
            summarize(rows, samples=50, frozen_fixed=fixed)["selected_fixed"], "fixed2"
        )
        for broken in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                validate_run(broken, manifest, {"records": len(broken)})
        del cfg["selected_fixed"]
        with self.assertRaisesRegex(ValueError, "frozen development"):
            validate_run(rows, manifest, {"records": 6})

    def test_clustered_statistics_and_correctness_gate(self):
        rows = []
        for i in range(6):
            for cap in (128, 512):
                for rep in range(3):
                    for conf, factor in [("target", 1), ("fixed2", 2), ("fixed4", 1.5)]:
                        rows.append(
                            dict(
                                prompt_id=f"development-qa-{i:02d}",
                                output_cap=cap,
                                repetition=rep,
                                configuration_id=conf,
                                correctness="pass",
                                generated_tokens=10,
                                generation_ms=100 / factor,
                                category="qa",
                                warm_ms=200,
                                load_ms=50,
                                peak_rss_bytes=1024,
                            )
                        )
        table = cells(rows)
        result = paired_summary(table, "fixed2", "target", samples=200)
        self.assertAlmostEqual(result["ratio"], 2)
        self.assertEqual(result["prompts"], 6)
        self.assertEqual(result["prompt_cap_cells"], 12)
        self.assertAlmostEqual(result["ci95"][0], 2)
        summary = summarize(rows, samples=200)
        self.assertEqual(summary["selected_fixed"], "fixed2")
        self.assertTrue(summary["q1_supported"])
        rows[0]["correctness"] = "pending"
        with self.assertRaises(ValueError):
            cells(rows)
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            cells([{**rows[1], "settings": {"diagnostic": True}}])
        for settings in ({"policy": "script"}, {"script": [0, 4]}):
            with self.assertRaisesRegex(ValueError, "diagnostic"):
                cells([{**rows[1], "settings": settings}])

    def test_replay_prediction_separates_fit_and_validation(self):
        records = []
        for i in range(6):
            for context in (128, 512):
                outcomes = {
                    str(k): {"rate": 0.01, "ms": 100, "tokens": 1}
                    for k in (0, 1, 2, 4, 6, 8)
                }
                winner = "2" if context < 256 else "8"
                outcomes[winner] = {"rate": 0.5, "ms": 10, "tokens": 5}
                outcomes["dynamic"] = {"rate": 0.01, "ms": 100, "tokens": 1}
                records.append(
                    dict(
                        prompt_id=f"development-qa-{i:02d}",
                        context=context,
                        acceptance=0.8,
                        outcomes=outcomes,
                        probe_ms=1,
                        checkpoint=context,
                    )
                )
        result = fit_validate(records, 0)
        self.assertTrue(result["q2_supported"])
        self.assertEqual(result["validation_prompts"], 3)
        # A predictor beating global fixed settings must still beat the existing
        # dynamic baseline before supporting a more complex controller.
        for row in records:
            row["outcomes"]["dynamic"]["rate"] = 1.0
        self.assertFalse(fit_validate(records, 0)["q2_supported"])
        for row in records:
            row["outcomes"]["dynamic"]["rate"] = 0.01
        for r in records:
            if int(r["prompt_id"].rsplit("-", 1)[1]) >= 3:
                r["outcomes"]["2"], r["outcomes"]["8"] = (
                    r["outcomes"]["8"],
                    r["outcomes"]["2"],
                )
        self.assertFalse(fit_validate(records, 0)["q2_supported"])
        for row in records:
            del row["outcomes"]["dynamic"]
        self.assertFalse(fit_validate(records, 0)["q2_supported"])

    def test_frozen_prompt_split_and_provenance(self):
        from bench.make_prompts import build

        rows = build()
        self.assertEqual(len(rows), 130)
        self.assertEqual(
            collections.Counter(r["split"] for r in rows),
            {"development": 40, "heldout": 90},
        )
        self.assertEqual(
            collections.Counter(r["category"] for r in rows if r["split"] == "heldout"),
            {"qa": 30, "code": 30, "summary": 30},
        )
        self.assertEqual(sum(r["pilot"] for r in rows), 18)
        self.assertEqual(len({r["sha256"] for r in rows}), 130)
