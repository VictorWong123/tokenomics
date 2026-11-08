import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from bench.common import sha256
from scripts.verify_experiment_evidence import unpack, retain


class FrozenArchiveTests(unittest.TestCase):
    def make_archive(self, root, variant="valid"):
        owner = {"run_id": "42466570831", "attempt": "2", "sha": "a" * 40}
        payload = b'{"state":"paused"}\n'
        manifest = {
            **owner,
            "files": {
                "status.json": {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            },
        }
        packed = root / "source.tar.gz"
        with tarfile.open(packed, "w:gz") as tar:
            entry = tarfile.TarInfo("status.json")
            entry.size = len(payload)
            tar.addfile(
                entry,
                io.BytesIO(
                    payload
                    if variant != "tampered"
                    else payload.replace(b"paused", b"failed")
                ),
            )
            if variant == "duplicate":
                tar.addfile(entry, io.BytesIO(payload))
            if variant == "unsafe":
                link = tarfile.TarInfo("../outside")
                link.type = tarfile.SYMTYPE
                link.linkname = "/tmp/outside"
                tar.addfile(link)
            if variant == "extra":
                other = tarfile.TarInfo("extra.txt")
                other.size = 1
                tar.addfile(other, io.BytesIO(b"x"))
            raw = json.dumps(manifest).encode()
            entry = tarfile.TarInfo("execution-manifest.json")
            entry.size = len(raw)
            tar.addfile(entry, io.BytesIO(raw))
        archive = root / "artifact.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.write(packed, "evidence.tar.gz")
            zipped.writestr(
                "report.json", json.dumps({**owner, "archive_sha256": sha256(packed)})
            )
        return archive

    def test_valid_archive_and_outer_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.make_archive(root)
            out = root / "checked"
            out.mkdir()
            report, manifest, evidence = unpack(archive, sha256(archive), out)
            self.assertEqual(report["run_id"], manifest["run_id"])
            self.assertEqual(
                (evidence / "status.json").read_text(), '{"state":"paused"}\n'
            )
            with self.assertRaisesRegex(ValueError, "ZIP digest"):
                unpack(archive, "0" * 64, out)

    def test_untrusted_members_and_changed_bytes_fail_closed(self):
        for variant, error in (
            ("unsafe", "Unsafe"),
            ("duplicate", "Duplicate"),
            ("extra", "Manifest member set"),
            ("tampered", "Member digest"),
        ):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive = self.make_archive(root, variant)
                out = root / "checked"
                out.mkdir()
                with self.assertRaisesRegex(ValueError, error):
                    unpack(archive, sha256(archive), out)
                self.assertFalse((out / "evidence/status.json").exists())

    def test_retained_bytes_and_canonical_receipt_are_immutable(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "status.json").write_text('{"state":"confirmation_complete"}\n')
            receipt = {
                "run_id": 42466570831,
                "attempt": 2,
                "source_sha": "a" * 40,
                "state": "confirmation_complete",
                "zip_sha256": "b" * 64,
                "archive_sha256": "c" * 64,
                "exact_comparisons": 162,
                "shutdown_confirmed": True,
                "independent_output_checks": True,
                "verifier_sha256": "first",
            }
            path = root / "receipt.json"
            path.write_text(json.dumps(receipt))
            with patch("scripts.verify_experiment_evidence.ROOT", root):
                retain(evidence, path)
                canonical = (
                    root
                    / "results/verification/codespace-confirmation-20261108/verification.json"
                )
                original = canonical.read_bytes()
                path.write_text(json.dumps({**receipt, "verifier_sha256": "later"}))
                retain(evidence, path)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(canonical.read_bytes(), original)
                (evidence / "status.json").write_text("changed")
                with self.assertRaisesRegex(ValueError, "immutable"):
                    retain(evidence, path)
