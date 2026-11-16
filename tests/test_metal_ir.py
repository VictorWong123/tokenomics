"""Text normalization checks; these are not GPU arithmetic tests."""

import unittest

from scripts.compare_metal_ir import KINDS, compare_ir, functions


def module(change=""):
    result = ['attributes #0 = { "unsafe-fp-math"="true" }']
    for kind in KINDS:
        for candidate in (False, True):
            name = f"kernel_mul_mv_{kind}_f32" + ("_interleaved" if candidate else "")
            grid, value, label = (
                ("%grid9", "%sum9", "end9") if candidate else ("%grid", "%sum", "end")
            )
            index = 1 if candidate else 0
            operation = "fmul" if candidate and change == "arithmetic" else "fadd"
            flag = "" if candidate and change == "fast" else "fast "
            offset = 8 if candidate and change == "address" else 4
            if candidate and change == "coordinate":
                index = 0
            result.append(
                f"""define void @{name}(<3 x i32> {grid}, ptr %weights) #0 {{
entry:
  %x = extractelement <3 x i32> {grid}, i32 {index}
  %pointer = getelementptr float, ptr %weights, i32 {offset}
  %loaded = load float, ptr %pointer, align 4, !tbaa !{99 if candidate else 7}
  {value} = {operation} {flag}float %loaded, 1.0
  br label %{label}
{label}:
  ret void
}}"""
            )
    return "\n".join(result)


class MetalIRTests(unittest.TestCase):
    def test_coordinate_permutation_and_ssa_renaming_match(self):
        result = compare_ir(module())
        self.assertEqual(len(functions(module())), 8)
        self.assertEqual(result["state"], "normalized_ir_matches")
        self.assertTrue(all(row["normalized_text_equal"] for row in result["pairs"]))
        self.assertFalse(result["gpu_execution"])

    def test_arithmetic_flags_coordinates_and_addresses_remain_significant(self):
        for change in ("arithmetic", "fast", "coordinate", "address"):
            with self.subTest(change=change):
                result = compare_ir(module(change))
                self.assertNotEqual(result["state"], "normalized_ir_matches")
                self.assertTrue(
                    all(row["state"] == "different" for row in result["pairs"])
                )

    def test_missing_functions_are_incomplete_not_equivalent(self):
        result = compare_ir("")
        self.assertEqual(result["state"], "diagnostic_incomplete_or_different")
        self.assertTrue(
            all(
                row["state"] == "missing_or_ambiguous_function"
                for row in result["pairs"]
            )
        )


if __name__ == "__main__":
    unittest.main()
