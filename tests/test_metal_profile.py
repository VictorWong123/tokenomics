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
    DIRECT_MARKER,
    check_direct_execution,
    direct_profile,
    INTERLEAVE_MARKER,
    Q8_GROUP_MARKER,
    Q8_SHARED_MARKER,
    q8_group_profile,
    q8_shared_profile,
    interleaved_profile,
    REFERENCE_MARKER,
    gate,
    metal_profile,
    verify_gate,
    greedy_profile,
)


class MetalProfileIdentityTests(unittest.TestCase):
    def test_model_specific_offload_and_q8_applicability_fail_closed(self):
        from analysis.metal_short_mv import screen_configuration
        from bench.model_identity import model_identity

        identity = model_identity("qwen25-15")
        self.assertEqual(identity["layers"], 28)
        self.assertEqual(identity["output_type"], "Q6_K")
        self.assertEqual(
            identity["sha256"],
            "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
        )
        row = {
            "system_info": CANDIDATE_MARKER
            + "; devices=MTL0; offloaded 29/29 layers to GPU",
            "settings": {"gpu_layers": 99},
        }
        metal_profile(row, True, model_id="qwen25-15")
        with self.assertRaises(ValueError):
            metal_profile(row, True)
        row["system_info"] = row["system_info"].replace("29/29", "25/25")
        with self.assertRaises(ValueError):
            metal_profile(row, True, model_id="qwen25-15")
        for kwargs in ({"group_q8": True}, {"shared_q8": True}):
            with self.assertRaisesRegex(ValueError, "no Q8 weights"):
                screen_configuration("retrieval", model_id="qwen25-15", **kwargs)
        config = screen_configuration(
            "retrieval", interleaved=True, model_id="qwen25-15"
        )
        self.assertEqual(config["model_id"], "qwen25-15")
        self.assertEqual(
            config["phase"], "metal_qwen25_15_interleaved_retrieval_comparison"
        )
        self.assertNotIn(
            "model_id", screen_configuration("retrieval", interleaved=True)
        )
        with self.assertRaisesRegex(ValueError, "Unsupported pinned model"):
            model_identity("unknown")

    def test_direct_sampler_requires_execution_evidence_and_isolated_identity(self):
        row = {
            "system_info": REFERENCE_MARKER
            + "; devices=MTL0; offloaded 25/25 layers to GPU"
            + "; TOKENOMICS_GREEDY_NO_TEMP=ON; "
            + DIRECT_MARKER
            + "; raw_greedy_eligible=1; raw_greedy_calls=3",
            "settings": {"gpu_layers": 99},
            "execution": {"cycles": 2, "verified_rows": 4},
        }
        direct_profile(row, True)
        for validator in (
            greedy_profile,
            metal_profile,
            interleaved_profile,
            q8_group_profile,
            q8_shared_profile,
        ):
            with self.assertRaises(ValueError):
                validator(row, True)
        for before, after in (
            ("raw_greedy_eligible=1", "raw_greedy_eligible=0"),
            ("raw_greedy_calls=3", "raw_greedy_calls=0"),
            ("raw_greedy_calls=3", "raw_greedy_calls=5"),
            ("raw_greedy_calls=3", ""),
            (DIRECT_MARKER, ""),
        ):
            with self.assertRaises(ValueError):
                check_direct_execution(
                    {**row, "system_info": row["system_info"].replace(before, after)}
                )
        empty = {**row, "execution": {"cycles": 0, "verified_rows": 0}}
        with self.assertRaises(ValueError):
            check_direct_execution(empty)
        empty["system_info"] = row["system_info"].replace(
            "raw_greedy_calls=3", "raw_greedy_calls=0"
        )
        direct_profile(empty, True)

    def test_shared_q8_and_sampler_profiles_cannot_be_mislabeled(self):
        row = {
            "system_info": CANDIDATE_MARKER
            + "; "
            + Q8_GROUP_MARKER
            + "; "
            + Q8_SHARED_MARKER
            + "; devices=MTL0; offloaded 25/25 layers to GPU",
            "settings": {"gpu_layers": 99},
        }
        q8_shared_profile(row, True)
        for validator in (
            metal_profile,
            interleaved_profile,
            q8_group_profile,
            greedy_profile,
        ):
            with self.assertRaises(ValueError):
                validator(row, True)
        row["system_info"] += "; TOKENOMICS_GREEDY_NO_TEMP=ON"
        with self.assertRaises(ValueError):
            q8_shared_profile(row, True)

    def test_q8_group_marker_excludes_other_kernel_variants(self):
        row = {
            "system_info": CANDIDATE_MARKER
            + "; devices=MTL0; offloaded 25/25 layers to GPU",
            "settings": {"gpu_layers": 99},
        }
        with self.assertRaises(ValueError):
            q8_group_profile(row, True)
        row["system_info"] += "; " + Q8_GROUP_MARKER
        q8_group_profile(row, True)
        for validator in (metal_profile, interleaved_profile):
            with self.assertRaises(ValueError):
                validator(row, True)
        row["system_info"] += "; " + INTERLEAVE_MARKER
        with self.assertRaises(ValueError):
            q8_group_profile(row, True)

    def test_interleave_is_explicit_and_excluded_from_other_profiles(self):
        row = {
            "system_info": CANDIDATE_MARKER
            + "; devices=MTL0; offloaded 25/25 layers to GPU",
            "settings": {"gpu_layers": 99},
        }
        with self.assertRaises(ValueError):
            interleaved_profile(row, True)
        row["system_info"] += "; " + INTERLEAVE_MARKER
        interleaved_profile(row, True)
        with self.assertRaises(ValueError):
            metal_profile(row, True)
        with self.assertRaises(ValueError):
            interleaved_profile(row, False)
        row["system_info"] = row["system_info"].replace(
            CANDIDATE_MARKER, REFERENCE_MARKER
        )
        with self.assertRaises(ValueError):
            interleaved_profile(row, False)
        row["system_info"] = row["system_info"].replace(INTERLEAVE_MARKER, "")
        interleaved_profile(row, False)

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
        self._screen_analysis("ngram")

    def test_retrieval_screen_checks_suffix_work_and_strong_baselines(self):
        self._screen_analysis("retrieval")

    def test_interleaved_retrieval_screen_has_distinct_frozen_identity(self):
        self._screen_analysis("retrieval", interleaved=True)

    def test_q8_group_retrieval_screen_keeps_distinct_profile_identity(self):
        self._screen_analysis("retrieval", group_q8=True)

    def test_q8_shared_retrieval_screen_keeps_distinct_profile_identity(self):
        self._screen_analysis("retrieval", shared_q8=True)

    def test_direct_sampler_screen_keeps_original_and_optimized_ordinary_controls(self):
        self._screen_analysis("retrieval", direct=True)

    def test_second_model_screen_binds_model_identity_and_geometry(self):
        self._screen_analysis("retrieval", interleaved=True, model_id="qwen25-15")

    def _screen_analysis(
        self,
        screen,
        interleaved=False,
        group_q8=False,
        shared_q8=False,
        direct=False,
        model_id="qwen25-05",
    ):
        from analysis.metal_short_mv import (
            SCREENS,
            analyze,
            kernel_variant,
            screen_configuration,
        )
        from bench.run import run
        from make_fixtures import create

        binary = Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bench").mkdir()
            (root / "configs").mkdir()
            config = json.loads((ROOT / "configs" / SCREENS[screen]).read_text())
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
            (root / "configs" / SCREENS[screen]).write_text(json.dumps(config))
            if (
                interleaved
                or group_q8
                or shared_q8
                or direct
                or model_id != "qwen25-05"
            ):
                with patch("analysis.metal_short_mv.ROOT", root):
                    config = screen_configuration(
                        screen, interleaved, group_q8, shared_q8, direct, model_id
                    )
            candidate_profile = kernel_variant(interleaved, group_q8, shared_q8, direct)
            model = root / "repeat.gguf"
            create(model, 100, ngram_cycle=True)
            # Distinct executable identity; same actual CPU runtime. This tests
            # the analysis pipeline, not the candidate's Metal behavior.
            wrapper = root / "candidate"
            wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(str(binary))} "$@"\n')
            wrapper.chmod(0o700)
            executables = {
                c["id"]: wrapper if c["kernel_profile"] == candidate_profile else binary
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
            ), patch(
                "analysis.metal_short_mv.model_identity",
                return_value={"sha256": sha256(model)},
            ), patch(
                "analysis.metal_short_mv.direct_profile"
                if direct
                else (
                    "analysis.metal_short_mv.q8_shared_profile"
                    if shared_q8
                    else (
                        "analysis.metal_short_mv.q8_group_profile"
                        if group_q8
                        else (
                            "analysis.metal_short_mv.interleaved_profile"
                            if interleaved
                            else "analysis.metal_short_mv.metal_profile"
                        )
                    )
                )
            ) as profile:
                result = analyze(
                    root / "screen",
                    screen=screen,
                    interleaved=interleaved,
                    group_q8=group_q8,
                    shared_q8=shared_q8,
                    direct=direct,
                    model_id=model_id,
                )
                self.assertEqual(result["records"], len(config["configurations"]) * 9)
                self.assertEqual(profile.call_count, result["records"])
                self.assertTrue(result["same_work_within_each_policy"])
                if direct:
                    self.assertEqual(result["kernel_variant"], "guarded-single-row")
                    self.assertEqual(result["sampler_variant"], "direct")
                    self.assertIn("direct-target/target", result["comparisons"])
                    self.assertIn("direct-fixed8/direct-target", result["comparisons"])
                rows = read_jsonl(root / "screen/requests.jsonl")
                rows[0]["generation_ms"] *= 0.5
                write_jsonl(root / "screen/requests.jsonl", rows)
                with self.assertRaises(ValueError):
                    analyze(
                        root / "screen",
                        screen=screen,
                        interleaved=interleaved,
                        group_q8=group_q8,
                        shared_q8=shared_q8,
                        direct=direct,
                        model_id=model_id,
                    )

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
