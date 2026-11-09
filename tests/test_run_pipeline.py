"""Timing and feasibility orchestration without launching full model pilots."""

import unittest
import pathlib
import tempfile
import os
import json
from bench.common import atomic_json, sha256


class RunPipelineTests(unittest.TestCase):
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
