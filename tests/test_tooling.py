import collections
import pathlib
import tempfile
import unittest
import os
import json
import math
from bench.run import balanced_blocks, first_mismatch
from bench.common import atomic_json, sha256
from analysis.report import cells, paired_summary, summarize, validate_run
from bench.replay import fit_validate


class ToolingTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("TOKENOMICS_TEST_BINARY"), "requires built fixture engine"
    )
    def test_paired_executables_are_used_and_frozen_on_resume(self):
        import subprocess
        import sys
        from unittest.mock import patch
        from bench.run import run

        exe = pathlib.Path(os.environ["TOKENOMICS_TEST_BINARY"])
        cfg = {
            "context": 128,
            "batch": 32,
            "threads": 1,
            "warmup_tokens": 2,
            "seed": 20261108,
            "output_caps": [4, 7],
            "repetitions": 1,
            "configurations": [
                {"id": "target", "policy": "target"},
                {"id": "alternate", "policy": "fixed", "k": 4, "drafter": "ngram"},
            ],
        }
        prompts = [{"id": "development-qa-00", "category": "qa", "prompt": "abcabcabc"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            alternate = root / "alternate"
            alternate.symlink_to(exe)
            mapping = {"target": exe, "alternate": alternate}
            original = subprocess.run
            launched = []

            def execute(args, **kwargs):
                launched.append(str(args[0]))
                return original(args, **kwargs)

            with patch("bench.run.subprocess.run", side_effect=execute):
                rows = run(
                    exe,
                    exe.parent / "fixtures/ngram-cycle.gguf",
                    None,
                    prompts,
                    cfg,
                    root / "run",
                    identity={"synthetic_fixture": True},
                    executables=mapping,
                )
            self.assertEqual(len(rows), 4)
            self.assertEqual(launched.count(str(alternate)), 2)
            frozen = json.loads((root / "run/run_manifest.json").read_text())["frozen"]
            self.assertEqual(
                frozen["configuration_binary_sha256"],
                {cid: sha256(path) for cid, path in mapping.items()},
            )
            with self.assertRaisesRegex(RuntimeError, "Resume refused"):
                run(
                    exe,
                    exe.parent / "fixtures/ngram-cycle.gguf",
                    None,
                    prompts,
                    cfg,
                    root / "run",
                    identity={"synthetic_fixture": True},
                    executables={**mapping, "alternate": pathlib.Path(sys.executable)},
                )
            with self.assertRaisesRegex(ValueError, "explicit executable"):
                run(exe, None, None, prompts, cfg, root / "bad", executables={})

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

    @unittest.skipUnless(
        os.getenv("TOKENOMICS_TEST_BINARY"), "requires built fixture engine"
    )
    def test_utf8_budget_boundaries_preserve_every_token_id(self):
        import subprocess

        exe = pathlib.Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        model = exe.parent / "fixtures/utf8-cycle.gguf"
        jobs = [
            {
                "prompt": "a b",
                "policy": policy,
                "k": 2,
                "script": [0, 2, 0, 4],
                "max_tokens": cap,
            }
            for policy in ("target", "fixed", "script")
            for cap in (0, 1, 2, 3, 4, 7)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "jobs.jsonl"
            path.write_text("".join(json.dumps(j) + "\n" for j in jobs))
            proc = subprocess.run(
                [
                    str(exe),
                    "--target",
                    str(model),
                    "--draft",
                    str(model),
                    "--jobs",
                    str(path),
                    "--threads",
                    "1",
                    "--context",
                    "64",
                    "--batch",
                    "32",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            rows = [json.loads(line) for line in proc.stdout.splitlines()]
            self.assertEqual(len(rows), len(jobs))
            for job, row in zip(jobs, rows):
                cap = job["max_tokens"]
                raw = (b"\xe2\x82\xac" * 3)[:cap]
                self.assertEqual(row["token_ids"], ([229, 133, 175] * 3)[:cap])
                self.assertEqual(row["generated_tokens"], cap)
                self.assertEqual(row["stop"], "budget")
                self.assertEqual(row["text"], raw.decode("utf-8", errors="replace"))
                self.assertEqual(row["text_utf8_repaired"], cap % 3 != 0)

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

    def test_model_gate_streams_and_stops_on_first_failure(self):
        import subprocess
        import sys
        from scripts.check_models import stream_process

        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "rows.jsonl"
            command = [
                sys.executable,
                "-c",
                "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('{\\\"token\\\": 1}\\n'); time.sleep(10)",
                str(out),
            ]
            seen = []

            def reject(row):
                seen.append(row)
                raise ValueError("first divergence")

            with (pathlib.Path(tmp) / "stderr").open("w") as log:
                with self.assertRaisesRegex(ValueError, "first divergence"):
                    stream_process(command, out, log, 2, reject)
            self.assertEqual(seen, [{"token": 1}])
            out.unlink()
            with (pathlib.Path(tmp) / "stderr").open("w") as log:
                with self.assertRaises(subprocess.TimeoutExpired):
                    stream_process(command, out, log, 0.5, seen.append)
            self.assertEqual(json.loads(out.read_text()), {"token": 1})

    def test_timing_only_stops_before_full_pilots(self):
        from argparse import Namespace
        from unittest.mock import patch
        from scripts import run_all

        binary = os.getenv("TOKENOMICS_TEST_BINARY")
        if not binary:
            self.skipTest("requires built fixture engine")
        binary = pathlib.Path(binary).resolve()
        fixture = binary.parent / "fixtures/target.gguf"
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "configs").mkdir()
            (root / "bench").mkdir()
            (root / "model-cache").mkdir()
            config = json.loads((run_all.ROOT / "configs/pilot-v1.json").read_text())
            atomic_json(root / "configs/pilot-v1.json", config)
            atomic_json(
                root / "configs/models-v1.json",
                {
                    "candidates": [
                        {
                            "id": "fixture",
                            "target": "t",
                            "draft": "d",
                            "min_available_gib": 0,
                        }
                    ]
                },
            )
            atomic_json(
                root / "model-cache/models.lock.json", {"models": {"t": {}, "d": {}}}
            )
            (root / "bench/prompts-v1.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "id": f"development-{cat}-00",
                            "category": cat,
                            "length_bucket": "long",
                            "pilot": True,
                            "split": "development",
                            "prompt": "a b",
                        }
                    )
                    + "\n"
                    for cat in ("qa", "code", "summary")
                )
            )
            args = Namespace(
                run_dir=root / "run",
                build_dir=binary.parent,
                backend="cpu",
                threads=1,
                fixtures_only=False,
                candidate="fixture",
                timing_only=True,
                max_hours=None,
                skip_ngram=False,
                model_dir=root / "model-cache",
            )
            real_run = run_all.subprocess.run

            def child_run(command, *positional, **kwargs):
                if pathlib.Path(command[0]).name == "ctest":
                    return run_all.subprocess.CompletedProcess(command, 0)
                return real_run(command, *positional, **kwargs)

            for ngram in (False, True):
                with self.subTest(ngram=ngram):
                    args.ngram_only = ngram
                    args.run_dir = root / ("ngram" if ngram else "model")
                    with patch.object(run_all, "ROOT", root), patch.object(
                        run_all, "download", return_value=fixture
                    ) as download, patch.object(
                        run_all,
                        "investigate",
                        side_effect=AssertionError("must not run Pilot B"),
                    ), patch.object(
                        run_all.subprocess, "run", side_effect=child_run
                    ):
                        # Skip only recursive ctest; run the actual tokenizer
                        # and screen with and without the draft model.
                        self.assertEqual(run_all.pipeline(args), 10)
                        self.assertEqual(download.call_count, 1 if ngram else 2)
                        self.assertEqual(
                            download.call_args.args[1:],
                            (args.model_dir, args.model_dir / "models.lock.json"),
                        )
                    point = "fixture-ngram" if ngram else "fixture"
                    timing = args.run_dir / point / "timing"
                    manifest = json.loads((timing / "run_manifest.json").read_text())
                    frozen = manifest["frozen"]
                    self.assertEqual(
                        frozen["draft_sha256"], None if ngram else sha256(fixture)
                    )
                    self.assertEqual(
                        json.loads((args.run_dir / "status.json").read_text())["state"],
                        "timing_complete",
                    )
                    self.assertEqual(
                        len((timing / "requests.jsonl").read_text().splitlines()),
                        18 if ngram else 21,
                    )
                    self.assertFalse((args.run_dir / point / "pilot-a").exists())
                    if ngram:
                        # Exercise the same snapshot validation invoked remotely,
                        # including the automatic no-draft ngram selection.
                        real_run(
                            [
                                str(pathlib.Path(os.sys.executable)),
                                "-m",
                                "bench.replay",
                                "--exe",
                                str(binary),
                                "--target",
                                str(fixture),
                                "--pilot-dir",
                                str(timing),
                                "--out",
                                str(root / "replay"),
                            ],
                            check=True,
                            stdout=run_all.subprocess.DEVNULL,
                        )
                        decision = json.loads(
                            (root / "replay/decision.json").read_text()
                        )
                        self.assertEqual(decision["schema_version"], 3)
                        self.assertIn("ngram", decision["dynamic_baseline"])

    def test_model_gate_rejects_divergence_and_stop_differences(self):
        from scripts.check_models import compare, variants

        expected = {
            "prompt_ids": [1],
            "token_ids": [2, 3],
            "stop": "budget",
            "generated_tokens": 2,
        }
        self.assertIsNone(compare(expected, dict(expected)))
        for field, value in (
            ("prompt_ids", [9]),
            ("token_ids", [2]),
            ("stop", "eog"),
            ("generated_tokens", 99),
        ):
            self.assertEqual(
                compare(expected, {**expected, field: value})["field"], field
            )
        configs = list(variants())
        self.assertEqual(len({c["configuration_id"] for c in configs}), 17)
        for cache in ("retain", "upstream"):
            self.assertEqual(
                {
                    c["k"]
                    for c in configs
                    if c.get("cache") == cache and c["policy"] == "fixed"
                },
                {1, 2, 4, 6, 8},
            )

    def test_model_gate_checks_cycle_and_cache_accounting(self):
        from scripts.check_models import check_cycles

        identity = {"prompt_id": "p", "configuration_id": "fixed4", "repetition": 0}
        row = {
            **identity,
            "prompt_ids": [1, 2],
            "generated_tokens": 2,
            "target_cached": 3,
            "draft_cached": 2,
        }
        cycle = {**identity, "committed": 2}
        check_cycles([row], [cycle])
        for changed in (
            {"generated_tokens": 3},
            {"target_cached": 4},
            {"draft_cached": 4},
        ):
            with self.assertRaises(RuntimeError):
                check_cycles([{**row, **changed}], [cycle])
        with self.assertRaises(RuntimeError):
            check_cycles([row], [{**cycle, "prompt_id": "unknown"}])

    def test_feasibility_decisions_do_not_confuse_access_with_science(self):
        from scripts.run_all import decision_state

        self.assertEqual(
            decision_state([{"status": "insufficient_available_memory"}])[1], 2
        )
        self.assertEqual(
            decision_state([{"q1_supported": False, "q2_supported": True}])[1], 22
        )
        self.assertEqual(
            decision_state([{"q1_supported": True, "q2_supported": False}])[1], 21
        )
        self.assertEqual(
            decision_state(
                [
                    {
                        "candidate": "model",
                        "q1_supported": False,
                        "q2_supported": False,
                    },
                    {"candidate": "ngram", "q1_supported": True, "q2_supported": True},
                ]
            )[1],
            20,
        )

    def test_balanced_order_and_determinism(self):
        prompts = [{"id": str(i)} for i in range(6)]
        confs = [{"id": str(i)} for i in range(3)]
        blocks = list(balanced_blocks(prompts, [128, 512], 3, confs, 123))
        self.assertEqual(
            blocks, list(balanced_blocks(prompts, [128, 512], 3, confs, 123))
        )
        self.assertEqual(len(blocks), 36)
        counts = collections.Counter(
            (c["id"], slot) for _, _, _, order in blocks for slot, c in enumerate(order)
        )
        self.assertEqual(set(counts.values()), {12})
        self.assertEqual(len({(p["id"], cap, r) for p, cap, r, _ in blocks}), 36)

    def test_first_divergence_including_length(self):
        self.assertIsNone(first_mismatch([1, 2], [1, 2]))
        self.assertEqual(first_mismatch([1, 2], [1, 3])["position"], 1)
        self.assertEqual(first_mismatch([1], [1, 2])["position"], 1)

    def test_atomic_artifact_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "record.json"
            atomic_json(p, {"x": 1})
            before = sha256(p)
            atomic_json(p, {"x": 2})
            self.assertNotEqual(before, sha256(p))
            self.assertFalse(p.with_suffix(".json.tmp").exists())

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

    @unittest.skipUnless(
        os.getenv("TOKENOMICS_TEST_BINARY"),
        "CMake supplies runtime binary for integration",
    )
    def test_planned_pause_resumes_complete_blocks_without_rerunning(self):
        from bench.run import run, RunPaused

        exe = pathlib.Path(os.environ["TOKENOMICS_TEST_BINARY"])
        model = exe.parent / "fixtures/target.gguf"
        cfg = {
            "output_caps": [8],
            "repetitions": 1,
            "seed": 5,
            "context": 64,
            "batch": 32,
            "threads": 1,
            "warmup_tokens": 2,
            "configurations": [
                {"id": "target", "policy": "target"},
                {"id": "fixed2", "policy": "fixed", "k": 2},
            ],
        }
        prompts = [
            {"id": f"development-qa-{i:02d}", "category": "qa", "prompt": "a"}
            for i in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            with self.assertRaises(RunPaused):
                run(exe, model, model, prompts, cfg, out, max_new_blocks=1)
            done = next(out.glob("blocks/*/complete.json"))
            original = done.read_bytes(), done.stat().st_mtime_ns
            self.assertFalse((out / "complete.json").exists())
            self.assertEqual(len((out / "requests.jsonl").read_text().splitlines()), 2)
            rows = run(exe, model, model, prompts, cfg, out)
            self.assertEqual(len(rows), 4)
            self.assertEqual((done.read_bytes(), done.stat().st_mtime_ns), original)
            self.assertEqual(len(list(out.glob("blocks/*/attempt-*"))), 2)

    @unittest.skipUnless(
        os.getenv("TOKENOMICS_TEST_BINARY"),
        "CMake supplies runtime binary for integration",
    )
    def test_incremental_resume_and_fingerprint(self):
        from bench.run import run

        exe = pathlib.Path(os.environ["TOKENOMICS_TEST_BINARY"])
        model = exe.parent / "fixtures/target.gguf"
        cfg = {
            "output_caps": [8],
            "repetitions": 1,
            "seed": 5,
            "context": 64,
            "batch": 32,
            "threads": 1,
            "warmup_tokens": 2,
            "configurations": [
                {"id": "target", "policy": "target"},
                {"id": "fixed2", "policy": "fixed", "k": 2},
            ],
        }
        prompts = [{"id": "development-qa-00", "category": "qa", "prompt": "a"}]
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d)
            rows = run(
                exe,
                model,
                model,
                prompts,
                cfg,
                out,
                identity={"synthetic_fixture": True},
            )
            self.assertEqual(len(rows), 2)
            done = next(out.glob("blocks/*/complete.json"))
            mtime = done.stat().st_mtime_ns
            self.assertEqual(
                run(
                    exe,
                    model,
                    model,
                    prompts,
                    cfg,
                    out,
                    identity={"synthetic_fixture": True},
                ),
                rows,
            )
            self.assertEqual(done.stat().st_mtime_ns, mtime)
            cfg["threads"] = 2
            with self.assertRaisesRegex(RuntimeError, "Resume refused"):
                run(
                    exe,
                    model,
                    model,
                    prompts,
                    cfg,
                    out,
                    identity={"synthetic_fixture": True},
                )

    @unittest.skipUnless(
        os.getenv("TOKENOMICS_TEST_BINARY"),
        "CMake supplies runtime binary for integration",
    )
    def test_failed_correctness_cannot_commit_block(self):
        from bench.run import run
        import sys

        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            fake = root / "fake-engine"
            fake.write_text(
                "#!"
                + sys.executable
                + "\n"
                + """import json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]))
rows=[]
for line in open(a['--jobs']):
 j=json.loads(line);j.update(prompt_ids=[1],token_ids=[1 if j['configuration_id']=='target' else 2],stop='budget');rows.append(j)
open(a['--out'],'w').write(''.join(json.dumps(j)+'\\n' for j in rows))
open(a['--cycles'],'w').close()
"""
            )
            fake.chmod(0o755)
            model = root / "model"
            model.write_text("fake")
            cfg = {
                "output_caps": [8],
                "repetitions": 1,
                "seed": 1,
                "context": 64,
                "batch": 32,
                "threads": 1,
                "warmup_tokens": 2,
                "configurations": [
                    {"id": "target", "policy": "target"},
                    {"id": "fixed2", "policy": "fixed", "k": 2},
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "token_mismatch"):
                run(
                    fake,
                    model,
                    model,
                    [{"id": "development-qa-00", "category": "qa", "prompt": "a"}],
                    cfg,
                    root / "run",
                )
            self.assertFalse(list((root / "run").glob("blocks/*/complete.json")))
            self.assertTrue(
                list((root / "run").glob("blocks/*/attempt-*/failure.json"))
            )
