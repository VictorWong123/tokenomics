"""Shared artifact encoding and hashing compatibility."""

import unittest
import pathlib
import tempfile
from bench.common import atomic_json, sha256


class ArtifactIOTests(unittest.TestCase):
    def test_jsonl_refactor_preserves_existing_artifact_bytes(self):
        from bench.common import write_jsonl, read_jsonl

        rows = [{"z": "\u03bb", "a": [1, True]}, {"value": None}]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "records.jsonl"
            write_jsonl(path, iter(rows))
            self.assertEqual(
                path.read_bytes(),
                b'{"z": "\\u03bb", "a": [1, true]}\n{"value": null}\n',
            )
            self.assertEqual(read_jsonl(path), rows)
            write_jsonl(path, rows, sort_keys=True)
            self.assertEqual(
                path.read_bytes(),
                b'{"a": [1, true], "z": "\\u03bb"}\n{"value": null}\n',
            )
            write_jsonl(path, [])
            self.assertEqual(path.read_bytes(), b"")

    def test_atomic_artifact_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "record.json"
            atomic_json(p, {"x": 1})
            before = sha256(p)
            atomic_json(p, {"x": 2})
            self.assertNotEqual(before, sha256(p))
            self.assertFalse(p.with_suffix(".json.tmp").exists())
