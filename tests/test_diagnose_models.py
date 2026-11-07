import copy
from types import SimpleNamespace
import unittest

from scripts.check_models import validate_record
from scripts.diagnose_models import probe_at


class DiagnosticContractTests(unittest.TestCase):
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
