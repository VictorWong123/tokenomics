"""Independent latest-edge search for causal linear transition composition."""

import itertools
import gzip
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from analysis.transition_replay import analyze, transition_draft


def reverse_search(sequence, cap, order):
    virtual = list(sequence)
    proposal = []
    for _ in range(cap):
        context = virtual[-order:]
        selected = None
        for end in range(len(sequence) - 1, order - 1, -1):
            if list(sequence[end - order : end]) == context:
                selected = sequence[end]
                break
        if selected is None:
            break
        proposal.append(selected)
        virtual.append(selected)
    return proposal


class TransitionReplayTest(unittest.TestCase):
    def test_exhaustive_frozen_observed_edges(self):
        cases = 0
        for size in range(1, 9):
            for sequence in itertools.product(range(3), repeat=size):
                for order, cap in itertools.product((1, 2), (0, 1, 4, 8)):
                    self.assertEqual(
                        transition_draft(sequence[:-1], sequence[-1], cap, order=order),
                        reverse_search(sequence, cap, order),
                    )
                    cases += 1
        self.assertEqual(cases, 78720)

    def test_composition_and_no_unobserved_edge(self):
        # Latest1->4 and4->1 compose a longer path than any contiguous suffix.
        self.assertEqual(
            transition_draft([1, 2, 3, 1, 4], 1, 8, order=1), [4, 1, 4, 1, 4, 1, 4, 1]
        )
        self.assertEqual(transition_draft([1, 2, 3], 4, 8, order=1), [])
        with self.assertRaises(ValueError):
            transition_draft([1], 2, 9, order=1)


class TransitionCollectionTest(unittest.TestCase):
    def test_legacy_fit_receipt_and_complete_report(self):
        output = [2, 1] * 32
        target = {
            "prompt_id": "fixture",
            "prompt_ids": [1, 2, 1],
            "token_ids": output,
            "stop": "budget",
            "rows": [
                {
                    "output_index": i,
                    "top_ids": [token] + [x for x in range(64) if x != token],
                }
                for i, token in enumerate(output)
            ],
        }
        # Original fit receipts predate the explicit collection_phase field.
        receipt = {"model_id": "fixture", "platform": {"machine": "fixture"}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "complete.json").write_text("{}")
            with patch(
                "analysis.transition_replay.load_collection",
                return_value=([target], receipt),
            ):
                report = analyze(root, root / "out")
            self.assertEqual(report["collection_phase"], "fit")
            self.assertEqual(len(report["totals"]), 14)
            self.assertEqual(
                report["computed_cycles"],
                sum(row["cycles"] for row in report["totals"]),
            )
            self.assertTrue(all(row["tokens"] == 64 for row in report["totals"]))
            self.assertEqual(
                set(json.loads((root / "out/complete.json").read_text())["sha256"]),
                {"report.json", "cycles.jsonl.gz"},
            )

    def test_priority_grid_preserves_strong_copy_without_ranking(self):
        output = [1, 2] * 32
        target = {
            "prompt_id": "fixture",
            "prompt_ids": [1, 2] * 5,
            "token_ids": output,
            "stop": "budget",
            "rows": [
                {
                    "output_index": i,
                    "top_ids": [token] + [x for x in range(64) if x != token],
                }
                for i, token in enumerate(output)
            ],
        }
        receipt = {"model_id": "fixture", "platform": {"machine": "fixture"}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "complete.json").write_text("{}")
            with patch(
                "analysis.transition_replay.load_collection",
                return_value=([target], receipt),
            ):
                report = analyze(root, root / "out", family="priority")
            self.assertEqual(len(report["totals"]), 16)
            aliases = [
                alias
                for point in report["work_frontier"]
                for alias in point["aliases"]
                if alias["id"].startswith("suffix5-priority-")
            ]
            self.assertTrue(aliases)
            self.assertTrue(
                all(alias["family"] == "suffix-priority" for alias in aliases)
            )
            traces = [
                json.loads(line)
                for line in gzip.decompress(
                    (root / "out/cycles.jsonl.gz").read_bytes()
                ).splitlines()
            ]
            priority = [
                row
                for row in traces
                if row["configuration"]["id"] == "suffix5-priority-guided3"
            ]
            self.assertTrue(priority)
            self.assertTrue(all(row["hint_output_index"] is None for row in priority))
            self.assertTrue(
                all(
                    row["hint_output_index"] is None
                    or row["hint_output_index"] == row["output_index"] - 1
                    for row in traces
                )
            )
            for cap in [1, 4, 8]:
                counts = {
                    row["id"]: row for row in report["totals"] if row["cap"] == cap
                }
                self.assertEqual(
                    counts["suffix5-priority-guided3"]["cycles"],
                    counts["suffix5"]["cycles"],
                )
                self.assertEqual(counts["suffix5-priority-guided3"]["wasted_rows"], 0)
