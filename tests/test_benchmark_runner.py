"""Balanced execution, executable identities and immutable resume."""

import collections
import pathlib
import tempfile
import unittest
import os
import json
from bench.run import balanced_blocks, first_mismatch
from bench.common import sha256


class BenchmarkRunnerTests(unittest.TestCase):
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


class DiagnosticSelection(unittest.TestCase):
    def test_scripted_diagnostic_cannot_launch_a_performance_condition(self):
        from unittest.mock import patch
        from bench.run import run

        for extra in ({"policy": "script"}, {"policy": "fixed", "script": [0, 4]}):
            cfg = {
                "configurations": [
                    {"id": "target", "policy": "target"},
                    {"id": "diagnostic", **extra},
                ]
            }
            with patch("bench.run.subprocess.run") as launch:
                with self.assertRaisesRegex(ValueError, "diagnostic"):
                    run(
                        pathlib.Path("not-an-executable"),
                        pathlib.Path("not-a-model"),
                        None,
                        [],
                        cfg,
                        pathlib.Path("unused"),
                    )
                launch.assert_not_called()
