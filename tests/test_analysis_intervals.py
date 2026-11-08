"""Degenerate bootstrap bounds render without changing reported statistics."""

import math
from pathlib import Path
import tempfile
import unittest
from analysis.report import draw_interval


class IntervalPlot(unittest.TestCase):
    def test_point_and_percentile_bound_can_differ_by_one_ulp(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i, value in enumerate((1.0, math.nextafter(1.0, 2.0))):
            interval = [math.nextafter(value, 2.0)] * 2
            before = list(interval)
            draw_interval(ax, value, i, interval)
            self.assertEqual(interval, before)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bounds.png"
            fig.savefig(output)
            self.assertGreater(output.stat().st_size, 100)
        plt.close(fig)

    def test_nonfinite_or_inverted_bounds_are_rejected(self):
        for interval in ([math.nan, 1.0], [0.0, math.inf], [2.0, 1.0]):
            with self.assertRaisesRegex(ValueError, "Invalid plotted"):
                draw_interval(None, 1.0, 0, interval)
