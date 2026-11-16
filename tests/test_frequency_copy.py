import itertools
import unittest

from analysis.frequency_copy import frequency_draft, frontier


def literal(observed, cap, minimum, support, confidence, maximum=32):
    for length in range(min(maximum, len(observed) - 1), minimum - 1, -1):
        continuations = [
            observed[start + length : start + length + cap]
            for start in reversed(range(len(observed) - length))
            if observed[start : start + length] == observed[-length:]
        ]
        if not continuations:
            continue
        proposal = []
        for size in range(1, cap + 1):
            prefixes = {
                tuple(seq[:size])
                for seq in continuations
                if len(seq) >= size and seq[: size - 1] == proposal
            }
            if not prefixes:
                break
            scored = []
            for prefix in prefixes:
                positions = [
                    index
                    for index, seq in enumerate(continuations)
                    if tuple(seq[:size]) == prefix
                ]
                scored.append((len(positions), -min(positions), prefix))
            count, _, best = max(scored)
            if (
                count < support
                or count * confidence[1] < len(continuations) * confidence[0]
            ):
                break
            proposal = list(best)
        if proposal:
            return proposal
    return []


class FrequencyCopyTests(unittest.TestCase):
    def test_matches_independent_prefix_enumeration(self):
        cases = 0
        for size in range(1, 9):
            for sequence in itertools.product(range(2), repeat=size):
                observed = list(sequence)
                for minimum, support, confidence, cap in itertools.product(
                    (1, 2, 3), (1, 2), ((1, 2), (1, 1)), (1, 2, 4)
                ):
                    actual = frequency_draft(
                        observed[:-1],
                        observed[-1],
                        cap,
                        minimum=minimum,
                        support=support,
                        confidence=confidence,
                    )
                    expected = literal(observed, cap, minimum, support, confidence)
                    self.assertEqual(actual, expected)
                    cases += 1
        self.assertEqual(cases, 18360)

    def test_joint_confidence_cannot_increase_after_censoring(self):
        observed = [1, 2, 7, 1, 2, 7, 1, 2]
        self.assertEqual(
            frequency_draft(
                observed[:-1],
                observed[-1],
                8,
                minimum=2,
                maximum=2,
                support=1,
                confidence=(1, 1),
            ),
            [7, 1, 2],
        )

    def test_recent_tie_and_longest_match_support_backoff(self):
        observed = [1, 2, 7, 1, 2, 8, 1, 2]
        self.assertEqual(
            frequency_draft(
                observed[:-1],
                observed[-1],
                1,
                minimum=2,
                maximum=2,
                support=1,
                confidence=(1, 2),
            ),
            [8],
        )
        self.assertEqual(
            frequency_draft(
                observed[:-1],
                observed[-1],
                1,
                minimum=2,
                maximum=2,
                support=1,
                confidence=(3, 4),
            ),
            [],
        )
        observed = [9, 1, 2, 7, 8, 1, 2, 7, 9, 1, 2]
        self.assertEqual(
            frequency_draft(
                observed[:-1],
                observed[-1],
                1,
                minimum=2,
                maximum=3,
                support=2,
                confidence=(1, 1),
            ),
            [7],
        )

    def test_pareto_aliases_are_deduplicated(self):
        cells = [
            dict(family="test", id=str(i), cap=1, cycles=c, wasted_rows=w)
            for i, (c, w) in enumerate([(10, 1), (10, 1), (11, 1), (12, 0), (9, 2)])
        ]
        points = frontier(cells)
        self.assertEqual(
            [(p["cycles"], p["wasted_rows"]) for p in points],
            [(9, 2), (10, 1), (12, 0)],
        )
        self.assertEqual(len(points[1]["aliases"]), 2)

    def test_empty_budget_and_invalid_settings(self):
        params = dict(minimum=1, support=1, confidence=(1, 1))
        self.assertEqual(frequency_draft([1, 1], 1, 0, **params), [])
        for changed in (dict(minimum=0), dict(support=0), dict(confidence=(2, 1))):
            with self.assertRaises(ValueError):
                frequency_draft([1, 1], 1, 1, **(params | changed))
