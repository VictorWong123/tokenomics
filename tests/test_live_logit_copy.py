"""Live replay agreement uses an independent ranked synthetic transformer."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from bench.common import sha256, write_jsonl
from scripts.check_logit_copy import check
from scripts.collect_logit_fit import collect, collection_protocol


@unittest.skipUnless(
    os.getenv("TOKENOMICS_LOGIT_BINARY") and os.getenv("TOKENOMICS_LOGIT_COPY_BINARY"),
    "requires collector and normal CPU binaries",
)
class LiveLogitCopyTests(unittest.TestCase):
    def test_live_correction_rows_match_all_frozen_validation_work(self):
        from make_fixtures import create

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "ranked-cycle.gguf"
            create(model, 100, logit_cycle=True)
            protocol, _ = collection_protocol("validation")
            corpus = root / "prompts.jsonl"
            write_jsonl(
                corpus,
                [
                    {"id": pid, "prompt": "abcdefghi" * 2}
                    for pid in json.loads(protocol.read_text())["prompt_ids"]
                ],
            )
            args = SimpleNamespace(
                binary=Path(os.environ["TOKENOMICS_LOGIT_BINARY"]).resolve(),
                model=model,
                model_id="qwen25-05",
                phase="validation",
                out=root / "collection",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                collect(args, expected_sha=sha256(model), corpus_path=corpus)
            live = SimpleNamespace(
                binary=Path(os.environ["TOKENOMICS_LOGIT_COPY_BINARY"]).resolve(),
                model=model,
                collection=args.out,
                corpus=corpus,
                out=root / "live",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                status = check(live, expected_sha=sha256(model))
            self.assertEqual(status["checked_comparisons"], 60)
            self.assertTrue(status["same_work_as_canonical_replay"])
            self.assertEqual(status["causal_hint_audit"], "passed")
            self.assertEqual(status["totals"]["ordinary-k0"]["drafted"], 0)
            self.assertGreater(status["totals"]["hint-m3-k64-f0-k8"]["accepted"], 0)
            self.assertTrue((live.out / "complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
