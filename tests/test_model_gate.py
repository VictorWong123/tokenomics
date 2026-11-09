"""Native generation correctness and Python gate failure handling."""

import pathlib
import tempfile
import unittest
import os
import json


class ModelGateTests(unittest.TestCase):
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
