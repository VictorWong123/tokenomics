import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.experiment_workload import pack


class ExperimentEvidenceTests(unittest.TestCase):
    def test_archive_hashes_and_rejects_unexpected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            results.mkdir()
            status = results / "status.json"
            status.write_text(json.dumps({"state": "failed"}))
            archive = root / "evidence.tgz"
            pack(results, archive, {"sha": "a" * 40})
            with tarfile.open(archive) as tar:
                manifest = json.load(tar.extractfile("execution-manifest.json"))
                self.assertEqual(set(manifest["files"]), {"status.json"})
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
