import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench.common import ROOT, atomic_json, sha256, source_fingerprint
from bench.cpu_profile import verified_packed_gate, study_configuration, profile_matches


class PackedControllerPreconditions(unittest.TestCase):
    def test_pending_or_invalid_proof_cannot_build_or_download(self):
        from scripts.controller_study import pipeline

        cfg = study_configuration(ROOT, "packed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg["trained_gate"] = "proof.json"
            config_path = root / "configs/controller-packed-study-v1.json"
            proof = dict(
                state="kernel_profile_passed",
                exact_comparisons=272,
                independent_output_checks=True,
                shutdown_confirmed=True,
                binary_sha256="a" * 64,
                screen_target_sha256="b" * 64,
            )
            for change in (
                None,
                {"state": "failed"},
                {"exact_comparisons": 271},
                {"independent_output_checks": False},
                {"shutdown_confirmed": False},
            ):
                atomic_json(root / "proof.json", {**proof, **(change or {})})
                cfg["trained_gate_sha256"] = (
                    "pending-trained-gate"
                    if change is None
                    else sha256(root / "proof.json")
                )
                atomic_json(config_path, cfg)
                with patch("scripts.controller_study.ROOT", root), patch(
                    "scripts.check_kernel_profile.command"
                ) as build, patch("scripts.controller_study.download") as download:
                    with self.assertRaisesRegex(RuntimeError, "verified trained"):
                        pipeline(root / "out", root / "models", cpu_profile="packed")
                    build.assert_not_called()
                    download.assert_not_called()
            cfg["trained_gate_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "provenance changed"):
                verified_packed_gate(root, cfg)

    def test_rebuild_must_match_gate_before_loading_models(self):
        from scripts.controller_study import pipeline

        cfg = study_configuration(ROOT, "packed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(
                root / "proof.json",
                dict(
                    state="kernel_profile_passed",
                    exact_comparisons=272,
                    independent_output_checks=True,
                    shutdown_confirmed=True,
                    binary_sha256="a" * 64,
                    screen_target_sha256="b" * 64,
                ),
            )
            cfg.update(
                trained_gate="proof.json",
                trained_gate_sha256=sha256(root / "proof.json"),
            )
            atomic_json(root / "configs/controller-packed-study-v1.json", cfg)
            binary = root / "build-cpu/tokenomics"
            binary.parent.mkdir()
            binary.write_text("different executable")
            with patch("scripts.controller_study.ROOT", root), patch(
                "scripts.controller_study.hardware", return_value={}
            ), patch("scripts.check_kernel_profile.command"), patch(
                "scripts.controller_study.download"
            ) as download:
                with self.assertRaisesRegex(RuntimeError, "executable differs"):
                    pipeline(root / "out", root / "models", cpu_profile="packed")
                download.assert_not_called()

    def test_profiles_are_explicit_and_mutually_exclusive(self):
        build = dict(
            values=dict(
                TOKENOMICS_CPU_BATCH_INVARIANT="OFF", TOKENOMICS_CPU_PACKED_VERIFY="ON"
            )
        )
        self.assertTrue(profile_matches(build, "packed"))
        self.assertFalse(profile_matches(build, "guarded"))
        with self.assertRaises(ValueError):
            study_configuration(ROOT, "unknown")

    def test_changed_target_is_rejected_before_controller_inference(self):
        from scripts.controller_study import pipeline

        cfg = study_configuration(ROOT, "packed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "build-cpu/tokenomics"
            binary.parent.mkdir()
            binary.write_text("fixture executable identity")
            target = root / "changed.gguf"
            target.write_text("fixture target identity")
            atomic_json(
                root / "proof.json",
                dict(
                    state="kernel_profile_passed",
                    exact_comparisons=272,
                    independent_output_checks=True,
                    shutdown_confirmed=True,
                    binary_sha256=sha256(binary),
                    screen_target_sha256="b" * 64,
                ),
            )
            cfg.update(
                trained_gate="proof.json",
                trained_gate_sha256=sha256(root / "proof.json"),
            )
            atomic_json(root / "configs/controller-packed-study-v1.json", cfg)
            (root / "bench").mkdir()
            (root / "bench/prompts-v1.jsonl").write_text("")
            with patch("scripts.controller_study.ROOT", root), patch(
                "scripts.controller_study.hardware", return_value={}
            ), patch("scripts.check_kernel_profile.command"), patch(
                "scripts.controller_study.build_configuration",
                return_value=dict(
                    values=dict(
                        TOKENOMICS_CPU_BATCH_INVARIANT="OFF",
                        TOKENOMICS_CPU_PACKED_VERIFY="ON",
                    )
                ),
            ), patch(
                "scripts.controller_study.available_memory", return_value=8 * 2**30
            ), patch(
                "scripts.controller_study.download", return_value=target
            ), patch(
                "scripts.controller_study.correctness_gate"
            ) as gate, patch(
                "scripts.controller_study.execute_study"
            ) as study:
                with self.assertRaisesRegex(RuntimeError, "target differs"):
                    pipeline(root / "out", root / "models", cpu_profile="packed")
                gate.assert_not_called()
                study.assert_not_called()


class ActualPackedControllerEvidence(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("TOKENOMICS_PACKED_TEST_BINARY"), "requires compiled packed profile"
    )
    def test_fresh_packed_calibration_and_profile_bound_evidence(self):
        from scripts.controller_study import execute_study, correctness_gate
        from scripts.verify_controller_evidence import check
        from scripts.diagnose_models import build_configuration

        exe = Path(os.environ["TOKENOMICS_PACKED_TEST_BINARY"]).resolve()
        model = exe.parent / "fixtures/ngram-cycle.gguf"
        cfg = study_configuration(ROOT, "packed")
        cfg.update(
            fit_prompt_ids=[cfg["fit_prompt_ids"][0]],
            validation_prompt_ids=[
                cfg["validation_prompt_ids"][0],
                cfg["validation_prompt_ids"][2],
            ],
            fit_output_caps=[64],
            fit_repetitions=1,
            output_caps=[16],
            repetitions=1,
            context=256,
            batch=32,
            threads=1,
            warmup_tokens=2,
            trained_gate="fixture-proof.json",
        )
        prompts = [
            dict(id=p, category=p.split("-")[1], split="development", prompt="a b")
            for p in cfg["fit_prompt_ids"] + cfg["validation_prompt_ids"]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Explicit synthetic harness prerequisite, not a trained-model receipt.
            proof = dict(
                state="kernel_profile_passed",
                exact_comparisons=272,
                independent_output_checks=True,
                shutdown_confirmed=True,
                binary_sha256=sha256(exe),
                screen_target_sha256=sha256(model),
            )
            atomic_json(root / "fixture-proof.json", proof)
            cfg["trained_gate_sha256"] = sha256(root / "fixture-proof.json")
            atomic_json(root / "configs/controller-packed-study-v1.json", cfg)
            atomic_json(
                root / "configs/controller-study-v1.json",
                study_configuration(ROOT, "guarded"),
            )
            out = root / "study"
            rows, _, _ = execute_study(
                exe, model, None, prompts, cfg, out, synthetic_fixture=True
            )
            prior = json.loads((out / "prior.json").read_text())
            self.assertEqual(
                prior["provenance"]["operating_point"]["binary_sha256"], sha256(exe)
            )
            self.assertIn(
                "TOKENOMICS_CPU_PACKED_VERIFY=ON",
                prior["provenance"]["operating_point"]["system_info"],
            )
            for drafter in ("model", "ngram"):
                correctness_gate(
                    exe,
                    model,
                    model if drafter == "model" else None,
                    prompts,
                    cfg,
                    out / ("gate-" + drafter),
                    drafter,
                )
            atomic_json(
                out / "controller-build.json",
                dict(binary_sha256=sha256(exe), configuration=build_configuration(exe)),
            )
            atomic_json(
                out / "status.json",
                dict(
                    state="controller_study_complete",
                    peak_rss_bytes=max(r["peak_rss_bytes"] for r in rows),
                ),
            )
            with patch("scripts.verify_controller_evidence.ROOT", root):
                kwargs = dict(
                    source=source_fingerprint()["revision"], synthetic_fixture=True
                )
                receipt = check(out, cpu_profile="packed", **kwargs)
                self.assertEqual(receipt["verified_validation_records"], 14)
                self.assertEqual(receipt["controller_gate_records"], 56)
                with self.assertRaisesRegex(ValueError, "execution profile"):
                    check(out, **kwargs)
                changed = copy.deepcopy(proof)
                changed["binary_sha256"] = "0" * 64
                atomic_json(root / "fixture-proof.json", changed)
                cfg["trained_gate_sha256"] = sha256(root / "fixture-proof.json")
                atomic_json(root / "configs/controller-packed-study-v1.json", cfg)
                with self.assertRaisesRegex(ValueError, "differs from"):
                    check(out, cpu_profile="packed", **kwargs)
