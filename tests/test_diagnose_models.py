import copy
from types import SimpleNamespace
import unittest

from scripts.check_models import validate_record
from scripts.diagnose_models import probe_at, replay_summary


class DiagnosticContractTests(unittest.TestCase):
    def test_matching_first_row_does_not_hide_later_divergence(self):
        reference = {"prompt_ids": [1], "token_ids": [7, 8, 10, 11, 12]}
        row = {
            "prompt_ids": [1, 7, 8],
            "token_ids": [10, 99, 12],
            "configuration_id": "prefix2-action4",
            "checkpoint": 2,
            "repetition": 0,
            "prompt_id": "p",
        }
        cycle = {
            "prompt_id": "p",
            "configuration_id": row["configuration_id"],
            "repetition": 0,
            "committed": 3,
            "actual_drafts": 4,
            "context_tokens": 3,
            "target_cached": 5,
            "diagnostic_logits": [
                {
                    "row": i,
                    "output_index": i,
                    "top_id": token,
                    "second_id": 0,
                    "top_logit": 2.0,
                    "second_logit": 1.0,
                }
                for i, token in enumerate(row["token_ids"])
            ],
        }
        summary = replay_summary(reference, row, [cycle])
        self.assertTrue(summary["probe_matches_emitted"])
        self.assertFalse(summary["matches_canonical_continuation"])
        self.assertEqual(summary["first_mismatch"]["position"], 1)
        self.assertTrue(summary["committed_rows"][1]["canonical_prefix"])
        self.assertFalse(summary["committed_rows"][2]["canonical_prefix"])
        wrong = {**row, "prompt_ids": [1, 7, 9]}
        with self.assertRaisesRegex(RuntimeError, "prefix differs"):
            replay_summary(reference, wrong, [cycle])

    def test_identity_and_settings_must_match(self):
        args = SimpleNamespace(context=2048, batch=512, threads=2, gpu_layers=0)
        job = dict(
            prompt_id="p",
            configuration_id="fixed4",
            repetition=0,
            policy="fixed",
            k=4,
            max_tokens=32,
            diagnostic_logits=True,
        )
        settings = dict(
            context=2048,
            batch=512,
            threads=2,
            gpu_layers=0,
            sampler="greedy-temperature-only",
            policy="fixed",
            k=4,
            cache="retain",
            drafter="model",
            confidence=0.5,
            max_tokens=32,
            max_cycles=0,
            script=[],
            diagnostic_logits=True,
        )
        row = dict(
            prompt_id="p",
            configuration_id="fixed4",
            repetition=0,
            output_cap=32,
            settings=settings,
        )
        validate_record(args, job, row)
        for key, value in [
            ("prompt_id", "other"),
            ("repetition", 1),
            ("output_cap", 128),
        ]:
            wrong = copy.deepcopy(row)
            wrong[key] = value
            with self.assertRaises(RuntimeError):
                validate_record(args, job, wrong)
        for key, value in [
            ("threads", 4),
            ("cache", "upstream"),
            ("diagnostic_logits", False),
        ]:
            wrong = copy.deepcopy(row)
            wrong["settings"][key] = value
            with self.assertRaises(RuntimeError):
                validate_record(args, job, wrong)

    def test_probe_uses_only_committed_matching_rows(self):
        probe = dict(
            row=0,
            output_index=8,
            top_id=10,
            second_id=11,
            top_logit=2.0,
            second_logit=1.9,
        )
        cycle = dict(
            prompt_id="p",
            configuration_id="c",
            repetition=0,
            committed=1,
            actual_drafts=4,
            context_tokens=20,
            target_cached=20,
            diagnostic_logits=[probe, {**probe, "row": 1, "output_index": 9}],
        )
        self.assertEqual(probe_at([cycle], "c", 0, 8, "p")["top_id"], 10)
        for index, prompt in [(9, "p"), (8, "other")]:
            with self.assertRaises(RuntimeError):
                probe_at([cycle], "c", 0, index, prompt)
        with self.assertRaises(RuntimeError):
            probe_at([cycle, cycle], "c", 0, 8, "p")
