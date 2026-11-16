"""Independent shortest-path checks for a future-aware diagnostic bound."""

from collections import deque
from itertools import product
import unittest

from analysis.retrieval_envelope import copy_options, envelope


def literal_shortest_path(prompt, output, cap, minimum):
    queue = deque([(0, 0)])
    visited = {0}
    while queue:
        index, cycles = queue.popleft()
        if index == len(output):
            return cycles
        observed = prompt + output[:index]
        destinations = {index + 1}
        # Enumerate copy origins and lengths directly, independently of the
        # production option generator and backward dynamic program.
        for start in range(len(observed)):
            for length in range(minimum, min(32, len(observed)) + 1):
                if start + length >= len(observed):
                    continue
                if observed[start : start + length] != observed[-length:]:
                    continue
                room = min(cap, len(output) - index - 1, len(observed) - start - length)
                for count in range(1, room + 1):
                    if (
                        observed[start + length : start + length + count]
                        == output[index : index + count]
                    ):
                        destinations.add(index + count + 1)
        for destination in destinations - visited:
            visited.add(destination)
            queue.append((destination, cycles + 1))
    raise AssertionError("Ordinary decoding must always reach the end")


class RetrievalEnvelopeTests(unittest.TestCase):
    def test_matches_independent_breadth_first_enumeration(self):
        cases = 0
        for prompt in product(range(2), repeat=2):
            for size in range(1, 6):
                for output in product(range(2), repeat=size):
                    for cap in (1, 2, 4, 6, 8):
                        for minimum in (1, 2):
                            result = envelope(list(prompt), list(output), cap, minimum)
                            self.assertEqual(
                                result["minimum_cycles"],
                                literal_shortest_path(
                                    list(prompt), list(output), cap, minimum
                                ),
                            )
                            self.assertEqual(result["verified_rows"], size)
                            self.assertEqual(result["wasted_rows"], 0)
                            cases += 1
        self.assertEqual(cases, 2480)

    def test_proposals_never_extend_past_the_observed_sequence(self):
        self.assertEqual(copy_options([1, 2, 1], 8, 1), {(), (2,), (2, 1)})
        self.assertEqual(copy_options([1, 2, 1], 0, 1), {()})
        self.assertEqual(copy_options([1], 8, 1), {()})
        for cap in (1, 2, 4, 6, 8):
            self.assertEqual(envelope([1], [2, 3, 4], cap, 1)["minimum_cycles"], 3)

    def test_larger_caps_and_shorter_minima_cannot_worsen_the_bound(self):
        prompt = [1, 2, 3, 1, 2, 4, 1, 2]
        output = [3, 1, 2, 4, 1, 2, 3, 1, 2, 4]
        previous = len(output)
        for cap in (1, 2, 4, 6, 8):
            result = envelope(prompt, output, cap, 1)["minimum_cycles"]
            self.assertLessEqual(result, previous)
            previous = result
            self.assertLessEqual(
                result, envelope(prompt, output, cap, 3)["minimum_cycles"]
            )


if __name__ == "__main__":
    unittest.main()
