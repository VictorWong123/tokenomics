"""Exercise the offline cycle audit's accounting and causality failures."""

import copy
import unittest

from analysis.research_archive import audit_cycles


class ResearchArchiveTest(unittest.TestCase):
    def test_work_and_causal_hint_rejections(self):
        identity = {
            "prompt_id": "fixture",
            "configuration_id": "guide",
            "repetition": 0,
        }
        sums = {"source_candidates": 0, "rank_checks": 0, "logits_examined": 0}
        row = {
            **identity,
            "prompt_ids": [1],
            "generated_tokens": 1,
            "target_cached": 1,
            "draft_cached": 0,
            "execution": {"cycles": 1, "drafted": 0, "accepted": 0, "verified_rows": 1},
            "settings": {"drafter": "logit-copy"},
            "logit_copy": sums,
        }
        cycle = {
            **identity,
            "cycle": 0,
            "committed": 1,
            "actual_drafts": 0,
            "accepted": 0,
            "logit_copy": {**sums, "hint_available": False, "hint_output_index": -1},
        }
        self.assertEqual(audit_cycles([row], [cycle]), 1)
        bad = copy.deepcopy(cycle)
        bad["accepted"] = 1
        with self.assertRaisesRegex(ValueError, "work/index"):
            audit_cycles([row], [bad])
        bad = copy.deepcopy(cycle)
        bad["logit_copy"].update(hint_available=True, hint_output_index=0)
        with self.assertRaisesRegex(ValueError, "Future/stale"):
            audit_cycles([row], [bad])
        bad = copy.deepcopy(cycle)
        bad["logit_copy"]["rank_checks"] = 1
        with self.assertRaisesRegex(ValueError, "telemetry"):
            audit_cycles([row], [bad])
