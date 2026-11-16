"""Independent prefix enumeration and causal guards for logit-guided chain replay."""

import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from analysis.logit_copy import (
    analyze,
    copy_candidate,
    guided_draft,
    import_collection,
    past_hint,
    replay_canonical,
)
from analysis.proposal_replay import suffix_draft
from bench.common import sha256, write_jsonl
from scripts.collect_logit_fit import collect


def enumerated(history, pending, cap, prior, minimum, cutoff, fallback):
    observed = list(history) + [pending]
    ranked = (
        {value: rank for rank, value in enumerate(prior[:cutoff]) if value != pending}
        if prior
        else {}
    )
    candidates = []
    for length in range(minimum, min(32, len(observed) - 1) + 1):
        for start in range(len(observed) - length):
            if observed[start : start + length] == observed[-length:]:
                boundary = start + length
                if observed[boundary] in ranked:
                    candidates.append(
                        (
                            (length, -ranked[observed[boundary]], boundary),
                            observed[boundary : boundary + cap],
                        )
                    )
    if candidates:
        return max(candidates, key=lambda row: row[0])[1]
    return suffix_draft(history, pending, cap, minimum=minimum) if fallback else []


class LogitCopyTests(unittest.TestCase):
    def test_exhaustive_selection_against_independent_prefix_enumeration(self):
        cases = 0
        for length in range(1, 8):
            for sequence in itertools.product(range(3), repeat=length):
                history, pending = list(sequence[:-1]), sequence[-1]
                prior = [pending] + [v for v in range(3) if v != pending]
                for minimum, cap in itertools.product((1, 3, 5), (0, 1, 4)):
                    self.assertEqual(
                        copy_candidate(history, pending, cap, minimum=minimum),
                        suffix_draft(history, pending, cap, minimum=minimum),
                    )
                    for cutoff, fallback in itertools.product((2, 3), (False, True)):
                        actual = guided_draft(
                            history,
                            pending,
                            cap,
                            prior,
                            minimum=minimum,
                            cutoff=cutoff,
                            fallback=fallback,
                        )
                        expected = enumerated(
                            history, pending, cap, prior, minimum, cutoff, fallback
                        )
                        self.assertEqual(actual, expected)
                        cases += 1
        self.assertEqual(cases, 118044)

    def test_first_cycle_has_no_hint_and_later_hint_is_strictly_past(self):
        class PastRows:
            def __init__(self):
                self.read = []

            def __getitem__(self, index):
                if index != 1:
                    raise AssertionError("future or wrong row read")
                self.read.append(index)
                return {"output_index": 1, "top_ids": [7, 3, 4]}

        rows = PastRows()
        self.assertIsNone(past_hint(rows, 0, 7))
        self.assertEqual(rows.read, [])
        self.assertEqual(past_hint(rows, 2, 7), [7, 3, 4])
        self.assertEqual(rows.read, [1])
        with self.assertRaisesRegex(ValueError, "Misaligned"):
            past_hint(rows, 2, 8)
        self.assertEqual(
            guided_draft([1, 2, 1], 2, 4, None, minimum=1, cutoff=8, fallback=False), []
        )
        with self.assertRaisesRegex(ValueError, "pending token"):
            guided_draft([1], 2, 1, [8, 1], minimum=1, cutoff=8, fallback=True)

    def test_early_eos_never_sets_the_draft_budget(self):
        seen = []

        def propose(history, pending, cap):
            seen.append(cap)
            return [7, 8, 9][:cap]

        target = {"prompt_ids": [1, 2, 3], "token_ids": [7], "stop": "eog"}
        counts, cycles = replay_canonical(target, 8, propose)
        self.assertEqual(seen, [8])
        self.assertEqual(
            (
                counts["cycles"],
                counts["drafted"],
                counts["accepted"],
                counts["wasted_rows"],
            ),
            (1, 3, 0, 3),
        )
        self.assertEqual(cycles[0]["committed"], 1)
        counts, cycles = replay_canonical(target, 1, lambda h, p, cap: [7])
        self.assertEqual(
            (counts["accepted"], cycles[0]["committed"], counts["wasted_rows"]),
            (1, 1, 1),
        )
        # Context capacity is known; it may legitimately clip the draft budget.
        context_target = {"prompt_ids": [1, 2, 3], "token_ids": [7], "stop": "context"}
        counts, cycles = replay_canonical(
            context_target, 8, lambda h, p, cap: [], context=3
        )
        self.assertEqual(cycles[0]["cap"], 0)

    def test_long_overlapping_sources_and_first_rank_exclusion(self):
        history = [1, 2, 3] * 30
        pending = 1
        for minimum in (1, 3, 5):
            self.assertEqual(
                copy_candidate(history, pending, 8, minimum=minimum),
                suffix_draft(history, pending, 8, minimum=minimum),
            )
        self.assertEqual(
            guided_draft(
                [5] * 40, 5, 8, [5, 7, 8], minimum=1, cutoff=3, fallback=False
            ),
            [],
        )
        with self.assertRaises(ValueError):
            copy_candidate([], 1, 9, minimum=1)


@unittest.skipUnless(
    os.getenv("TOKENOMICS_LOGIT_BINARY"), "requires CPU rank collector"
)
class LogitCopyBundleTests(unittest.TestCase):
    def test_validation_phase_uses_only_the_six_frozen_prompts_and_ten_cells(self):
        from make_fixtures import create
        from scripts.collect_logit_fit import collection_protocol

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "fixture.gguf"
            create(model, 100, ngram_cycle=True)
            protocol, _ = collection_protocol("validation")
            ids = json.loads(protocol.read_text())["prompt_ids"]
            corpus = root / "corpus.jsonl"
            write_jsonl(corpus, [{"id": pid, "prompt": "abcdefghi" * 2} for pid in ids])
            args = SimpleNamespace(
                binary=Path(os.environ["TOKENOMICS_LOGIT_BINARY"]).resolve(),
                model=model,
                model_id="qwen25-05",
                phase="validation",
                out=root / "collected",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = collect(args, expected_sha=sha256(model), corpus_path=corpus)
            self.assertEqual(status["ranking_rows"], 384)
            log = root / "workflow.log"
            log.write_text(output.getvalue())
            receipt = json.loads((args.out / "manifest.json").read_text())
            import_collection(
                log,
                root / "imported",
                "https://github.com/VictorWong123/tokenomics/actions/runs/1",
                receipt["source"]["revision"],
                allow_synthetic=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = analyze(root / "imported", root / "replay")
            self.assertEqual(result["collection_phase"], "validation")
            self.assertEqual(len(result["totals"]), 10)
            self.assertEqual(len(result["per_prompt"]), 60)
            self.assertEqual(sum(c["family"] == "guided" for c in result["totals"]), 3)
            ordinary = next(c for c in result["totals"] if c["id"] == "ordinary")
            self.assertEqual((ordinary["cycles"], ordinary["wasted_rows"]), (384, 0))

    def test_public_log_transport_replay_and_mutation_rejection(self):
        from make_fixtures import create

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "fixture.gguf"
            create(model, 100, ngram_cycle=True)
            corpus = root / "corpus.jsonl"
            ids = ["development-qa-01", "development-code-01", "development-summary-01"]
            write_jsonl(corpus, [{"id": pid, "prompt": "abcdefghi" * 2} for pid in ids])
            args = SimpleNamespace(
                binary=Path(os.environ["TOKENOMICS_LOGIT_BINARY"]).resolve(),
                model=model,
                model_id="qwen25-05",
                out=root / "collected",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                collect(args, expected_sha=sha256(model), corpus_path=corpus)
            log = root / "workflow.log"
            log.write_text(output.getvalue())
            receipt = json.loads((args.out / "manifest.json").read_text())
            run_url = "https://github.com/VictorWong123/tokenomics/actions/runs/1"
            with self.assertRaisesRegex(ValueError, "synthetic/learned"):
                import_collection(
                    log, root / "rejected", run_url, receipt["source"]["revision"]
                )
            missing = root / "missing.log"
            lines = output.getvalue().splitlines()
            first_data = next(
                i for i, line in enumerate(lines) if line.startswith("LOGIT_FIT_DATA=")
            )
            missing.write_text(
                "\n".join(lines[:first_data] + lines[first_data + 1 :]) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "Missing or unexpected"):
                import_collection(
                    missing,
                    root / "incomplete",
                    run_url,
                    receipt["source"]["revision"],
                    allow_synthetic=True,
                )
            imported = root / "imported"
            import_collection(
                log,
                imported,
                run_url,
                receipt["source"]["revision"],
                allow_synthetic=True,
            )
            self.assertEqual(
                (args.out / "data.jsonl").read_bytes(),
                (imported / "data.jsonl").read_bytes(),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = analyze(imported, root / "replay")
            self.assertTrue(result["synthetic_fixture"])
            self.assertEqual(len(result["totals"]), 48)
            self.assertFalse(result["runtime_selected"])
            (imported / "data.jsonl").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "collection changed"):
                analyze(imported, root / "mutated")


if __name__ == "__main__":
    unittest.main()
