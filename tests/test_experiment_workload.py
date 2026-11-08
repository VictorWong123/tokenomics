import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.experiment_workload import pack


class ExperimentEvidenceTests(unittest.TestCase):
    def test_diagnostic_report_needs_no_timing_rows_or_environment_file(self):
        from scripts.experiment_workload import diagnostic_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnosis = dict(
                state="profile_diagnosis_complete",
                diagnostic_only=True,
                both_original_histories_reproduced=True,
                performance_summary_produced=False,
            )
            (root / "diagnosis.json").write_text(json.dumps(diagnosis))
            result = diagnostic_report(root)
            self.assertEqual(result["profile_diagnosis"], diagnosis)
            self.assertNotIn("measurements", result)
            self.assertNotIn("environment", result)
            diagnosis["performance_summary_produced"] = True
            (root / "diagnosis.json").write_text(json.dumps(diagnosis))
            with self.assertRaisesRegex(RuntimeError, "Invalid profile diagnosis"):
                diagnostic_report(root)

    def test_packed_controller_reads_its_validation_grid_and_diagnostics_have_none(
        self,
    ):
        from scripts.experiment_workload import measurement_directory

        root = Path("results")
        self.assertEqual(
            measurement_directory("controller-packed", root, "unused"),
            root / "validation",
        )
        self.assertEqual(
            measurement_directory("controller", root, "unused"), root / "validation"
        )
        with self.assertRaisesRegex(ValueError, "Diagnostic"):
            measurement_directory("profile-diagnosis", root, "unused")

    def test_tagged_measurements_are_explicit_and_preflight_before_power_on(self):
        import yaml
        from bench.common import ROOT

        workflow = yaml.load(
            (ROOT / ".github/workflows/codespace-experiment.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        self.assertEqual(
            workflow["on"]["push"],
            {"tags": ["tokenomics-profiles-*", "tokenomics-controller-packed-*"]},
        )
        self.assertEqual(
            workflow["concurrency"]["group"], "tokenomics-dedicated-codespace"
        )
        steps = workflow["jobs"]["initialize"]["steps"]
        checks = [
            i for i, s in enumerate(steps) if "verified_packed_gate" in s.get("run", "")
        ]
        start = next(i for i, s in enumerate(steps) if s.get("id") == "lifecycle")
        self.assertEqual(len(checks), 1)
        self.assertLess(checks[0], start)
        self.assertEqual(steps[checks[0]]["if"], "github.event_name == 'push'")
        self.assertNotIn("GH_TOKEN", steps[checks[0]].get("env", {}))
        workload = workflow["jobs"]["workload"]
        self.assertEqual(workload["needs"], "initialize")
        self.assertEqual(workload["if"], "needs.initialize.outputs.claimed == 'true'")
        environment = next(
            s["env"] for s in workload["steps"] if s.get("id") == "fixtures"
        )
        self.assertNotIn("GH_TOKEN", environment)
        self.assertIn("controller-packed", environment["TOKENOMICS_EXPERIMENT_PHASE"])
        self.assertIn("'profiles'", environment["TOKENOMICS_EXPERIMENT_PHASE"])

    def test_archive_hashes_and_rejects_unexpected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            results.mkdir()
            status = results / "status.json"
            status.write_text(json.dumps({"state": "failed"}))
            model_log = results / "candidate.stderr.log"
            model_log.write_text("model load diagnostics\n")
            archive = root / "evidence.tgz"
            pack(results, archive, {"sha": "a" * 40})
            with tarfile.open(archive) as tar:
                manifest = json.load(tar.extractfile("execution-manifest.json"))
                self.assertEqual(
                    set(manifest["files"]), {"status.json", "candidate.stderr.log"}
                )
                self.assertEqual(
                    manifest["files"]["status.json"]["bytes"], status.stat().st_size
                )
            (results / "weights.gguf").write_text("excluded")
            with self.assertRaises(RuntimeError):
                pack(results, archive, {})

    def test_symlinks_and_missing_status_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            results.mkdir()
            with self.assertRaises(RuntimeError):
                pack(results, root / "evidence.tgz", {})
            (results / "status.json").symlink_to(root / "outside.json")
            with self.assertRaises(RuntimeError):
                pack(results, root / "evidence.tgz", {})
