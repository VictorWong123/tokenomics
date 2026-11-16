"""Gate orchestration tests; CPU fixtures do not validate Metal arithmetic."""

import json
import os
from pathlib import Path
import shlex
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from bench.common import ROOT, read_jsonl, sha256, write_jsonl
from scripts.check_metal_profile import (
    CANDIDATE_MARKER,
    REFERENCE_MARKER,
    gate,
    metal_profile,
    verify_gate,
    greedy_profile,
)


class MetalProfileIdentityTests(unittest.TestCase):
    def test_legacy_receipt_preserves_its_original_coverage(self):
        archive_path = (
            ROOT
            / "results/verification/greedy-sampler-local-20261116/synthetic-gate.tar.gz"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with tarfile.open(archive_path) as archive:
                for member in archive.getmembers():
                    parts = Path(member.name).parts
                    self.assertTrue(
                        member.isfile() and len(parts) == 2 and parts[0] == "gate"
                    )
                    target = root / "gate" / parts[1]
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
            result = verify_gate(root / "gate")
            self.assertEqual(result["gate_version"], 1)
            self.assertEqual(result["candidate_cases"], 330)

    def test_greedy_profile_requires_guarded_metal_and_distinct_sampler_marker(self):
        row = {
            "system_info": REFERENCE_MARKER
            + "; devices=MTL0; offloaded 25/25 layers to GPU",
            "settings": {"gpu_layers": 99},
        }
        greedy_profile(row, False)
        with self.assertRaises(ValueError):
            greedy_profile(row, True)
        row["system_info"] += "; TOKENOMICS_GREEDY_NO_TEMP=ON"
        greedy_profile(row, True)
        with self.assertRaises(ValueError):
            greedy_profile(row, False)
        row["system_info"] += CANDIDATE_MARKER
        with self.assertRaises(ValueError):
            greedy_profile(row, True)

    def test_rejects_cpu_fallback_and_conflicting_profile(self):
        row = {
            "system_info": CANDIDATE_MARKER
            + "; devices=MTL0; offloaded 25/25 layers to GPU",
            "settings": {"gpu_layers": 99},
        }
        metal_profile(row, True)
        for bad in (
            {**row, "system_info": row["system_info"].replace("MTL0", "CPU")},
            {**row, "system_info": row["system_info"] + REFERENCE_MARKER},
            {**row, "settings": {"gpu_layers": 0}},
        ):
            with self.assertRaises(ValueError):
                metal_profile(bad, True)
        with self.assertRaises(ValueError):
            metal_profile(row, False)


@unittest.skipUnless(os.getenv("TOKENOMICS_TEST_BINARY"), "requires runtime executable")
class MetalGateOrchestrationTests(unittest.TestCase):
    def test_screen_analysis_audits_real_cpu_fixture_blocks(self):
        from analysis.metal_short_mv import analyze
        from bench.run import run
        from make_fixtures import create

        binary = Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bench").mkdir()
            (root / "configs").mkdir()
            config = json.loads(
                (ROOT / "configs/metal-short-mv-screen-v1.json").read_text()
            )
            config["context"] = 64
            config["threads"] = 1
            prompts = [
                {
                    "id": pid,
                    "prompt": "abcdefghi" * 2,
                    "split": "development",
                    "category": "synthetic",
                }
                for pid in config["prompt_ids"]
            ]
            write_jsonl(root / "bench/prompts-v1.jsonl", prompts)
            (root / "configs/metal-short-mv-screen-v1.json").write_text(
                json.dumps(config)
            )
            model = root / "repeat.gguf"
            create(model, 100, ngram_cycle=True)
            # Distinct executable identity; same actual CPU runtime. This tests
            # the analysis pipeline, not the candidate's Metal behavior.
            wrapper = root / "candidate"
            wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(str(binary))} "$@"\n')
            wrapper.chmod(0o700)
            executables = {
                c["id"]: wrapper if c["kernel_profile"] == "short-mv" else binary
                for c in config["configurations"]
            }
            run(
                binary,
                model,
                None,
                prompts,
                config,
                root / "screen",
                executables=executables,
                identity={"synthetic_fixture": True},
            )
            with patch("analysis.metal_short_mv.ROOT", root), patch(
                "analysis.metal_short_mv.MODEL_SHA256", sha256(model)
            ), patch("analysis.metal_short_mv.metal_profile") as profile:
                result = analyze(root / "screen")
                self.assertEqual(result["records"], 72)
                self.assertEqual(profile.call_count, 72)
                self.assertTrue(result["same_work_within_each_policy"])
                rows = read_jsonl(root / "screen/requests.jsonl")
                rows[0]["generation_ms"] *= 0.5
                write_jsonl(root / "screen/requests.jsonl", rows)
                with self.assertRaises(ValueError):
                    analyze(root / "screen")

    def test_cross_binary_gate_retains_success_and_first_failure(self):
        from make_fixtures import create

        binary = Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bench").mkdir()
            ids = [
                "development-code-03",
                "development-qa-01",
                "development-code-01",
                "development-summary-01",
            ]
            write_jsonl(
                root / "bench/prompts-v1.jsonl",
                [{"id": pid, "prompt": "abcdefghi" * 2} for pid in ids],
            )
            model = root / "repeat.gguf"
            create(model, 100, ngram_cycle=True)

            def cpu_fixture(row, candidate):
                self.assertEqual(row["settings"]["gpu_layers"], 0)

            with patch("scripts.check_metal_profile.ROOT", root):
                result = gate(
                    binary,
                    binary,
                    model,
                    root / "passed",
                    context=64,
                    threads=1,
                    gpu_layers=0,
                    profile_validator=cpu_fixture,
                )
                self.assertEqual(result["candidate_cases"], 450)
                self.assertGreater(result["coverage"]["accepted"], 0)
                self.assertEqual(result["state"], "passed")
                self.assertEqual(verify_gate(root / "passed")["candidate_cases"], 450)
                saved = root / "passed/candidate.jsonl"
                original = saved.read_bytes()
                saved.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "Gate payload changed"):
                    verify_gate(root / "passed")
                saved.write_bytes(original)
                suffix_result = gate(
                    binary,
                    binary,
                    model,
                    root / "suffix-passed",
                    context=512,
                    threads=1,
                    gpu_layers=0,
                    profile_validator=cpu_fixture,
                    drafter="suffix",
                )
                self.assertEqual(suffix_result["candidate_cases"], 600)
                self.assertEqual(
                    verify_gate(root / "suffix-passed")["candidate_cases"], 600
                )
                boundary_jobs = read_jsonl(
                    root / "suffix-passed/boundary-reference.jobs.jsonl"
                )
                self.assertEqual(
                    {
                        len(job["prefix_tokens"])
                        for job in boundary_jobs
                        if job["prompt_id"].startswith("kv-padding-")
                    },
                    {248, 255, 256, 504, 511},
                )
                self.assertGreater(suffix_result["coverage"]["accepted"], 0)

                # Simulate a genuine output divergence at the comparison boundary.
                def divergent(row, candidate):
                    cpu_fixture(row, candidate)
                    if candidate:
                        row["token_ids"][0] += 1

                with self.assertRaisesRegex(RuntimeError, "Cross-profile divergence"):
                    gate(
                        binary,
                        binary,
                        model,
                        root / "failed",
                        context=64,
                        threads=1,
                        gpu_layers=0,
                        profile_validator=divergent,
                    )
            status = json.loads((root / "failed/status.json").read_text())
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["first_mismatch"]["mismatch"]["field"], "token_ids")
            self.assertTrue((root / "failed/smoke-candidate.cycles.jsonl").exists())
            self.assertFalse((root / "failed/candidate.jobs.jsonl").exists())
            self.assertTrue(read_jsonl(root / "failed/correctness.jsonl"))


if __name__ == "__main__":
    unittest.main()
