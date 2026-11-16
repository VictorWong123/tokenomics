import tempfile
from pathlib import Path
import unittest

from analysis.proposal_replay import (
    CPU_FIT,
    analyze_fit,
    extract_mac_fit,
    replay,
    suffix_draft,
    upstream_ngram,
)
from bench.replay import infer_drafter, replay_checkpoints


class ProposalReplayTests(unittest.TestCase):
    def test_checkpoint_schedule_keeps_full_action_room(self):
        self.assertEqual(replay_checkpoints({}, 64), [32])
        self.assertEqual(
            replay_checkpoints({"replay_checkpoints": [9, 16, 32, 48]}, 64),
            [9, 16, 32, 48],
        )
        self.assertEqual(replay_checkpoints({"replay_checkpoints": [9, 16]}, 25), [9])
        for schedule in ([], [8], [True], [16, 9], [9, 9], "32", list(range(9, 26))):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                replay_checkpoints({"replay_checkpoints": schedule}, 64)

    def test_replay_passes_only_committed_history(self):
        prompt, output = [9, 8, 7], list(range(1, 18))
        visits = []

        def proposals(history, pending, cap):
            index = len(history) + 1 - len(prompt)
            self.assertEqual(history + [pending], prompt + output[:index])
            visits.append(index)
            return []

        counts, cycles = replay(prompt, output, 8, proposals)
        self.assertEqual(visits, list(range(17)))
        self.assertEqual(counts["cycles"], 17)
        self.assertEqual(counts["wasted_rows"], 0)
        self.assertIsNone(counts["required_fixed_cost_fraction"])
        counts, cycles = replay([1] * 6, [1] * 17, 8, lambda h, p, cap: [p] * cap)
        self.assertEqual(counts["cycles"], 2)
        self.assertEqual(counts["verified_rows"], 17)
        self.assertEqual(cycles[-1]["cap"], 7)

    def test_suffix_overlap_and_minimum_match(self):
        self.assertEqual(suffix_draft([1] * 5, 1, 8, minimum=5), [1])
        self.assertEqual(suffix_draft([1] * 4, 1, 8, minimum=5), [])
        self.assertEqual(upstream_ngram([1] * 20, 1, 8), [])
        # Earliest full-context occurrence is valid in suffix retrieval.
        history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4]
        self.assertEqual(suffix_draft(history, 5, 4, minimum=5), [6, 7, 8, 9])

    def test_archived_fit_baselines_are_reproduced(self):
        with tempfile.TemporaryDirectory() as tmp:
            mac = analyze_fit(extract_mac_fit(Path(tmp)), "test-mac")
        cpu = analyze_fit(CPU_FIT, "test-cpu")
        for source in (mac, cpu):
            self.assertEqual(source["baseline_policy_checks"], 15)
            self.assertEqual(source["baseline_detailed_cycle_checks"], 0)
            selected = next(
                r
                for r in source["totals"]
                if r["strategy"] == "suffix5" and r["cap"] == 1
            )
            self.assertEqual(selected["wasted_rows"], 0)
            self.assertGreater(selected["accepted"], 0)

    def test_replay_drafter_identity_is_not_inferred_from_absent_model(self):
        for drafter in ("ngram", "suffix", "model"):
            frozen = {
                "draft_sha256": "abc" if drafter == "model" else None,
                "config": {
                    "configurations": [
                        {"policy": "target"},
                        {"policy": "fixed", "drafter": drafter},
                    ]
                },
            }
            self.assertEqual(infer_drafter(frozen), drafter)
        frozen["draft_sha256"] = None
        with self.assertRaises(ValueError):
            infer_drafter(frozen)
        frozen["config"]["configurations"] += [{"policy": "fixed", "drafter": "suffix"}]
        with self.assertRaises(ValueError):
            infer_drafter(frozen)


if __name__ == "__main__":
    unittest.main()
