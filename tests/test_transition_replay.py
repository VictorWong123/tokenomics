"""Independent latest-edge search for causal linear transition composition."""

import itertools
import unittest

from analysis.transition_replay import transition_draft


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
