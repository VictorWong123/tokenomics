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


class PartialControllerRetention(unittest.TestCase):
    def test_partial_attempt_cannot_become_final_canonical_point(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "status.json").write_text('{"state":"paused"}\n')
            receipt = dict(
                run_id=42492089188,
                attempt=1,
                source_sha="a" * 40,
                state="paused",
                zip_sha256="b" * 64,
                archive_sha256="c" * 64,
                exact_comparisons=7,
                shutdown_confirmed=True,
                independent_output_checks=True,
            )
            path = root / "receipt.json"
            path.write_text(json.dumps(receipt))
            with patch("scripts.verify_experiment_evidence.ROOT", root):
                retain(evidence, path)
                retained = root / "results/verification/codespace-controller-20261108"
                self.assertTrue(
                    (retained / "verification-attempt-1-paused.json").exists()
                )
                self.assertFalse((retained / "verification.json").exists())
                path.write_text(
                    json.dumps(
                        {
                            **receipt,
                            "attempt": 2,
                            "state": "controller_study_complete",
                            "exact_comparisons": 308,
                        }
                    )
                )
                retain(evidence, path)
                final = (retained / "verification.json").read_bytes()
                path.write_text(json.dumps(receipt))
                retain(evidence, path)
                self.assertEqual((retained / "verification.json").read_bytes(), final)

    def test_partial_second_controller_profile_remains_attempt_specific(self):
        from unittest.mock import patch
        from scripts import verify_experiment_evidence as verifier

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "status.json").write_text('{"state":"paused"}\n')
            path = root / "receipt.json"
            path.write_text(
                json.dumps(
                    dict(
                        run_id=999,
                        attempt=1,
                        source_sha="a" * 40,
                        state="paused",
                        zip_sha256="b" * 64,
                        archive_sha256="c" * 64,
                        exact_comparisons=7,
                        shutdown_confirmed=True,
                        independent_output_checks=True,
                    )
                )
            )
            with patch.object(verifier, "ROOT", root), patch.dict(
                verifier.EXPECTED, {999: "a" * 40}
            ), patch.dict(
                verifier.FOLDERS, {999: "fixture-packed-controller"}
            ), patch.dict(
                verifier.CONTROLLER_PROFILES, {999: "packed"}
            ):
                verifier.retain(evidence, path)
            retained = root / "results/verification/fixture-packed-controller"
            self.assertTrue((retained / "verification-attempt-1-paused.json").is_file())
            self.assertFalse((retained / "verification.json").exists())


class HostedLifecycleArtifact(unittest.TestCase):
    def test_only_bounded_digest_checked_finalizer_json_is_read(self):
        from scripts.verify_experiment_evidence import lifecycle_archive

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant in ("valid", "extra", "oversized", "wrong-mode"):
                path = root / (variant + ".zip")
                body = dict(
                    mode="initialize" if variant == "wrong-mode" else "finalize",
                    passed=True,
                    shutdown_confirmed=True,
                    run_id="42492089188",
                    attempt="1",
                    sha="a" * 40,
                )
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(
                        "report.json",
                        " " * 65537 if variant == "oversized" else json.dumps(body),
                    )
                    if variant == "extra":
                        archive.writestr("unexpected.txt", "rejected")
                if variant == "valid":
                    self.assertEqual(lifecycle_archive(path, sha256(path)), body)
                    with self.assertRaisesRegex(ValueError, "digest"):
                        lifecycle_archive(path, "0" * 64)
                else:
                    with self.assertRaises(ValueError):
                        lifecycle_archive(path, sha256(path))


class ScreenStatistics(unittest.TestCase):
    def test_regenerated_screen_rejects_changed_statistics_without_writing_raw(self):
        import shutil
        from bench.common import ROOT
        from scripts.verify_experiment_evidence import verify_screen_statistics

        root = ROOT / "results/verification/codespace-ngram-20261108"
        # Existing immutable trained records; this test starts no inference.
        point = root / "qwen25-05-3-q4-ngram/timing"
        archived = root / "analysis"
        original = (point / "requests.jsonl").read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            for name in ("measurements.json", "stage-costs.json"):
                shutil.copy2(archived / name, destination / name)
            verified = verify_screen_statistics(point, destination)
            self.assertFalse(verified["synthetic_fixture"])
            path = destination / "measurements.json"
            changed = json.loads(path.read_text())
            changed["geomean_ratios_vs_target"]["fixed1"] += 0.1
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "statistics differ"):
                verify_screen_statistics(point, destination)
        self.assertEqual((point / "requests.jsonl").read_bytes(), original)
