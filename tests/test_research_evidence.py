"""Reject damaged or unsealed research evidence before preserving it."""

import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.recover_research_evidence import digest, import_log, validate


def archive(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name, value in files.items():
            zipped.writestr(name, value)
    return stream.getvalue()


class ResearchEvidenceTest(unittest.TestCase):
    def test_seal_and_payload_damage(self):
        raw = b'{"tokens":[1,2,3]}\n'
        files = {
            "live/data.jsonl": raw,
            "live/complete.json": json.dumps({"sha256": {"data.jsonl": digest(raw)}}),
        }
        data = archive(files)
        self.assertEqual(
            validate(data)["seals"]["live/complete.json"]["checked_payloads"], 1
        )
        for name, value in [
            ("live/data.jsonl", b"changed"),
            ("../outside.json", b"{}"),
            ("weights.gguf", b"data"),
        ]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate(archive({**files, name: value}))
        with self.assertRaises(ValueError):
            validate(archive({"data.jsonl": raw}))

    def test_transport_identity_and_corruption(self):
        data = archive(
            {
                "data.jsonl": b"{}",
                "complete.json": json.dumps({"sha256": {"data.jsonl": digest(b"{}")}}),
            }
        )
        receipt = {
            "run_id": 1,
            "run_url": "https://github.com/VictorWong123/tokenomics/actions/runs/1",
            "source_commit": "a" * 40,
            "conclusion": "success",
            "artifact_name": "fixture",
            "zip_sha256": digest(data),
            "bytes": len(data),
            "chunks": 1,
            "validation": validate(data),
        }
        chunk = {"run_id": 1, "index": 0, "base64": base64.b64encode(data).decode()}

        def lines(r, c):
            return (
                "prefix RESEARCH_EVIDENCE_CHUNK="
                + json.dumps(c)
                + "\nprefix RESEARCH_EVIDENCE_RECEIPT="
                + json.dumps(r)
                + "\n"
            )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.recover_research_evidence.RUNS", {1: ("fixture", "a" * 40)}
        ):
            root = Path(tmp)
            log = root / "job.log"
            log.write_text(lines(receipt, chunk))
            import_log(log, root / "good")
            self.assertEqual((root / "good/1.zip").read_bytes(), data)
            for i, (r, c) in enumerate(
                [
                    ({**receipt, "zip_sha256": "0" * 64}, chunk),
                    ({**receipt, "source_commit": "b" * 40}, chunk),
                    ({**receipt, "validation": {}}, chunk),
                    (receipt, {**chunk, "base64": chunk["base64"][:-4] + "AAAA"}),
                ]
            ):
                log.write_text(lines(r, c))
                with self.subTest(i=i), self.assertRaises(ValueError):
                    import_log(log, root / f"bad-{i}")
                self.assertFalse((root / f"bad-{i}").exists())
