"""Canonical ranking collection is a diagnostic, never benchmark evidence."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from bench.common import sha256, write_jsonl
from scripts.collect_logit_fit import collect


@unittest.skipUnless(
    os.getenv("TOKENOMICS_LOGIT_BINARY"), "requires opt-in CPU ranking collector"
)
class LogitCollectionTests(unittest.TestCase):
    def test_stable_ties_canonical_outputs_and_hashes(self):
        from make_fixtures import create

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "fixture.gguf"
            create(model, 259)
            ids = ("development-qa-01", "development-code-01", "development-summary-01")
            corpus = root / "prompts.jsonl"
            write_jsonl(corpus, [{"id": pid, "prompt": "a b"} for pid in ids])
            args = SimpleNamespace(
                binary=Path(os.environ["TOKENOMICS_LOGIT_BINARY"]).resolve(),
                model=model,
                model_id="qwen25-05",
                out=root / "fit",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                status = collect(args, expected_sha=sha256(model), corpus_path=corpus)
            self.assertEqual(status["ranking_rows"], 192)
            self.assertTrue(status["synthetic_fixture"])
            rows = [
                json.loads(line)
                for line in (args.out / "data.jsonl").read_text().splitlines()
            ]
            for row in rows:
                self.assertEqual(row["token_ids"], [259] * 64)
                for rank in row["rows"]:
                    self.assertEqual(rank["top_ids"], [259] + list(range(63)))
            complete = json.loads((args.out / "complete.json").read_text())
            self.assertTrue(
                all(
                    sha256(args.out / name) == digest
                    for name, digest in complete["sha256"].items()
                )
            )
            args.out = root / "wrong-model"
            with self.assertRaisesRegex(ValueError, "Wrong pinned model"):
                collect(args, corpus_path=corpus)
            self.assertFalse((args.out / "complete.json").exists())


if __name__ == "__main__":
    unittest.main()
