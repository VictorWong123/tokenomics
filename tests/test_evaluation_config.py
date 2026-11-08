"""Pure configuration checks; these priors never launch inference or claim timings."""

import copy
import json
import unittest
from bench.common import ROOT, read_jsonl
from scripts.evaluate_controller import prepare


class EvaluationConfig(unittest.TestCase):
    def setUp(self):
        self.template = json.loads((ROOT / "configs/evaluation-v1.json").read_text())
        self.prompts = read_jsonl(ROOT / "bench/prompts-v1.jsonl")
        self.prior = dict(
            prior_id="unit-config-only-never-executed",
            synthetic=False,
            provenance=dict(
                split="development",
                fit_prompt_ids=["development-qa-01"],
                selected_fixed="fixed2",
                operating_point=dict(
                    drafter="ngram", cache="retain", context=2048, batch=512, threads=2
                ),
            ),
        )

    def test_full_grid_and_development_baseline_are_frozen(self):
        selected, cfg = prepare(self.prior, "a" * 64, self.prompts, self.template)
        self.assertEqual(len(selected), 90)
        self.assertEqual(cfg["repetitions"], 5)
        self.assertEqual(cfg["output_caps"], [128, 512])
        self.assertEqual(cfg["threads"], 2)
        self.assertEqual(cfg["selected_fixed"], "fixed2")
        self.assertEqual(len(cfg["configurations"]) * len(selected) * 2 * 5, 6300)
        self.assertEqual(
            next(c for c in cfg["configurations"] if c["id"] == "fixed2")["k"], 2
        )
        self.assertTrue(all(not p["id"].startswith("development-") for p in selected))
        self.assertEqual(self.template["configurations"], [])

    def test_incomplete_shortened_or_leaking_claims_are_rejected(self):
        for prompts, template, prior in (
            (self.prompts[:-1], self.template, self.prior),
            (self.prompts, {**self.template, "repetitions": 4}, self.prior),
            (self.prompts, {**self.template, "output_caps": [64, 128]}, self.prior),
            (self.prompts, self.template, {**self.prior, "synthetic": True}),
        ):
            with self.subTest(template=template.get("output_caps")), self.assertRaises(
                ValueError
            ):
                prepare(prior, "a" * 64, prompts, template)
        prior = copy.deepcopy(self.prior)
        prior["provenance"]["fit_prompt_ids"] = [
            next(p["id"] for p in self.prompts if p["split"] == "heldout")
        ]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            prepare(prior, "a" * 64, self.prompts, self.template)
