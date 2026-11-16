"""Historical development-validation replay; never executes a language model."""

import unittest

from analysis.proposal_validation import evaluate
from bench.common import ROOT


class FrozenProposalValidationTests(unittest.TestCase):
    def test_reproduces_recorded_ngram_work_and_retains_rejections(self):
        result = evaluate(
            ROOT
            / "results/verification/codespace-controller-20261108/attempt-2/validation",
            "cpu",
        )
        self.assertEqual(result["baseline_work_checks"], 72)
        self.assertEqual(len(result["cells"]), 36)
        self.assertEqual(len(result["comparisons"]), 12)
        total = result["totals"]["suffix5-cap1"]
        self.assertEqual(total["tokens"], 1152)
        self.assertEqual(total["verified_rows"], total["cycles"] + total["drafted"])
        self.assertEqual(total["tokens"], total["cycles"] + total["accepted"])
        self.assertGreater(total["wasted_rows"], 0)
        self.assertLess(total["wasted_rows"], result["totals"]["ngram1"]["wasted_rows"])
        self.assertGreater(total["cycles"], result["totals"]["ngram1"]["cycles"])


if __name__ == "__main__":
    unittest.main()
