import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.q4_pair import analyze
from bench.common import ROOT, atomic_json
from bench.run import run
from scripts.experiment_workload import measurement_directory


class Q4PairStudyTests(unittest.TestCase):
    def test_kernel_include_participates_in_resume_fingerprint(self):
        from bench.common import source_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "CMakeLists.txt").write_text("project(test)")
            (root / "requirements.txt").write_text("")
            kernel = root / "src/kernel.inc"
            kernel.write_text("first implementation")
            with patch("bench.common.ROOT", root), patch(
                "bench.common.command", return_value={"output": "fixed-revision"}
            ):
                before = source_fingerprint()
                kernel.write_text("changed implementation")
                after = source_fingerprint()
            self.assertEqual(before["revision"], after["revision"])
            self.assertNotEqual(
                before["owned_source_sha256"], after["owned_source_sha256"]
            )

    def test_workflow_selects_the_fixed_study(self):
        import yaml

        workflow = yaml.load(
            (ROOT / ".github/workflows/codespace-experiment.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        self.assertIn(
            "q4-pair", workflow["on"]["workflow_dispatch"]["inputs"]["phase"]["options"]
        )
        self.assertEqual(
            measurement_directory("q4-pair", Path("evidence"), "unused"),
            Path("evidence/q4-pair"),
        )
        cfg = json.loads((ROOT / "configs/q4-pair-screen-v1.json").read_text())
        self.assertEqual(
            len(cfg["prompt_ids"])
            * len(cfg["configurations"])
            * len(cfg["output_caps"])
            * cfg["repetitions"],
            72,
        )

    @unittest.skipUnless(
        os.environ.get("TOKENOMICS_Q4_PAIR_TEST_BINARY")
        and os.environ.get("TOKENOMICS_TEST_BINARY"),
        "requires both guarded and Q4 pair binaries",
    )
    def test_actual_two_executable_grid_and_changed_protocol_rejection(self):
        from make_fixtures import create
        from scripts.verify_q4_pair_evidence import verify_summary

        guarded = Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        paired = Path(os.environ["TOKENOMICS_Q4_PAIR_TEST_BINARY"]).resolve()
        cfg = json.loads((ROOT / "configs/q4-pair-screen-v1.json").read_text())
        # Short synthetic prompts retain the exact frozen settings/condition grid.
        prompts = [
            {
                "id": pid,
                "split": "development",
                "category": "code",
                "length_bucket": "short",
                "prompt": "abcdefghiabcdefghiabcdefghi",
            }
            for pid in cfg["prompt_ids"]
        ]
        exes = {
            c["id"]: paired if c["kernel_profile"] == "q4-pair" else guarded
            for c in cfg["configurations"]
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            model = out / "repeat-q4.gguf"
            create(model, 259, k_quantized=True, ngram_cycle=True, width_override=512)
            run(
                guarded,
                model,
                None,
                prompts,
                cfg,
                out,
                0,
                {"synthetic_fixture": True},
                executables=exes,
            )
            summary = analyze(out)
            self.assertEqual(summary["records"], 72)
            self.assertTrue(summary["cross_profile_exact"])
            self.assertTrue(summary["synthetic_fixture"])
            self.assertGreater(summary["work_totals"]["pair-fixed1"]["drafted"], 0)
            self.assertGreater(summary["work_totals"]["pair-fixed8"]["accepted"], 0)
            self.assertEqual(verify_summary(out), summary)
            altered_summary = {**summary, "records": 71}
            summary_path = out / "analysis/summary.json"
            atomic_json(summary_path, altered_summary)
            with self.assertRaisesRegex(ValueError, "summary differs"):
                verify_summary(out)
            self.assertEqual(json.loads(summary_path.read_text()), altered_summary)
            atomic_json(summary_path, summary)
            manifest_path = out / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            altered = copy.deepcopy(manifest)
            altered["frozen"]["config"]["seed"] += 1
            atomic_json(manifest_path, altered)
            with self.assertRaises(ValueError):
                analyze(out)
