import json
import pathlib
import tarfile
import tempfile
import unittest

from scripts.model_workload import DIAGNOSTIC_FILES, pack


class ModelEvidenceTests(unittest.TestCase):
    def test_failed_attempt_retains_only_allowlisted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            results = root / "results"
            results.mkdir()
            (results / "status.json").write_text(json.dumps({"state": "failed"}))
            (results / "secret.env").write_text("excluded")
            (results / "weights.gguf").write_text("excluded")
            archive = root / "evidence.tgz"
            pack(results, archive, {"sha": "a" * 40, "run_id": "1", "attempt": "1"})
            with tarfile.open(archive) as tar:
                self.assertEqual(
                    set(tar.getnames()), {"status.json", "execution-manifest.json"}
                )

    def test_diagnostic_completion_cannot_claim_correctness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            results = root / "results"
            (results / "diagnostic").mkdir(parents=True)
            (results / "status.json").write_text(
                json.dumps({"state": "diagnostic_complete"})
            )
            for name in DIAGNOSTIC_FILES:
                (results / "diagnostic" / name).write_text("{}")
            status = {
                "state": "diagnostic_complete",
                "synthetic_fixture": False,
                "correctness_supported": False,
                "full_comparisons": [{}] * 14,
            }
            (results / "diagnostic/status.json").write_text(json.dumps(status))
            pack(results, root / "evidence.tgz", {})
            status["correctness_supported"] = True
            (results / "diagnostic/status.json").write_text(json.dumps(status))
            with self.assertRaises(RuntimeError):
                pack(results, root / "evidence.tgz", {})

    def test_false_pass_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            results = root / "results"
            results.mkdir()
            (results / "status.json").write_text(json.dumps({"state": "passed"}))
            with self.assertRaises(RuntimeError):
                pack(results, root / "evidence.tgz", {})
            (results / "status.json").unlink()
            (root / "outside.json").write_text(json.dumps({"state": "failed"}))
            (results / "status.json").symlink_to(root / "outside.json")
            with self.assertRaises(RuntimeError):
                pack(results, root / "evidence.tgz", {})


if __name__ == "__main__":
    unittest.main()
