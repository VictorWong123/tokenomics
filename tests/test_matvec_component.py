import copy
import unittest

from analysis.matvec_component import PROFILES, ROWS, SHAPES, summarize


def fixture():
    records = []
    for profile in PROFILES:
        for block in range(3):
            for kind, width, outputs in SHAPES:
                for rows in ROWS:
                    base = dict(
                        profile=profile,
                        block=block,
                        type=kind,
                        width=width,
                        outputs=outputs,
                        rows=rows,
                        backend="MTL0",
                        synthetic_fixture=False,
                        weight_bytes=100,
                        weight_fnv1a64=1,
                    )
                    records.append(
                        dict(
                            base,
                            state="bitwise_passed",
                            compared_values=outputs * rows,
                            input_fnv1a64=rows,
                            output_fnv1a64=rows + 10,
                        )
                    )
                    for trial in range(16):
                        records.append(
                            dict(
                                base,
                                state="timing",
                                trial=trial,
                                repetitions=8,
                                batch_us=10 if profile == "short-mv" else 5,
                                serial_graph_us=20,
                            )
                        )
    return records


class MatvecComponentTests(unittest.TestCase):
    def test_component_ratios_are_separate_from_serial_control(self):
        result = summarize(fixture())
        self.assertEqual(result["trial_pairs"], 2880)
        self.assertEqual(len(result["cells"]), 20)
        values = result["cells"][0]["profiles"]
        self.assertEqual(values["interleaved"]["median_ratio_vs_short_mv"], 2)
        self.assertEqual(
            values["interleaved"]["blocks"][0]["median_serial_graph_ratio"], 4
        )

    def test_missing_duplicate_failure_and_data_changes_rejected(self):
        rows = fixture()
        bad = [rows[:-1], rows + [rows[0]]]
        changed_timing = copy.deepcopy(rows)
        changed_timing[1]["weight_fnv1a64"] = 999
        bad.append(changed_timing)
        for field, value in (
            ("output_fnv1a64", 999),
            ("input_fnv1a64", 999),
            ("weight_fnv1a64", 999),
            ("state", "bitwise_mismatch"),
            ("backend", "CPU"),
            ("synthetic_fixture", True),
        ):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            bad.append(changed)
        for changed in bad:
            with self.assertRaises(ValueError):
                summarize(changed)

    def test_invalid_timings_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            rows = fixture()
            rows[1]["batch_us"] = value
            with self.assertRaises(ValueError):
                summarize(rows)
