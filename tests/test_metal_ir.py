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


def forwarding_module(*, wrong_mask=False, wrong_callee=False, raw_use=False):
    functions = []
    for kind in KINDS:
        for candidate in (False, True):
            name = f"kernel_mul_mv_{kind}_f32" + ("_interleaved" if candidate else "")
            grid = "%9" if candidate else "%0"
            callee = "different_impl" if candidate and wrong_callee else "same_impl"
            instructions = []
            argument = grid
            if candidate:
                mask = "0, i32 1" if wrong_mask else "1, i32 0"
                instructions.append(
                    f"  %10 = shufflevector <3 x i32> {grid}, <3 x i32> undef, <3 x i32> <i32 {mask}, i32 2>"
                )
                argument = grid if raw_use else "%10"
            instructions += [
                f"  tail call void @{callee}(<3 x i32> {argument})",
                "  ret void",
            ]
            functions.append(
                f"define void @{name}(<3 x i32> {grid}) {{\n"
                + "\n".join(instructions)
                + "\n}"
            )
    functions.append("define void @same_impl(<3 x i32> %position) {\n  ret void\n}")
    return "\n".join(functions)


class MetalIRTests(unittest.TestCase):
    def test_additional_kernels_are_explicitly_outside_the_comparison(self):
        reference = forwarding_module()
        candidate = (
            reference + "\ndefine void @new_unverified_kernel() {\n  ret void\n}"
        )
        result = compare_ir(candidate, reference_module=reference)
        self.assertEqual(result["state"], "normalized_ir_matches")
        self.assertEqual(
            result["uncompared_additional_functions"], ["new_unverified_kernel"]
        )

    def test_forwarding_wrappers_cancel_only_the_exact_grid_swap(self):
        result = compare_ir(forwarding_module())
        self.assertEqual(result["state"], "normalized_ir_matches")
        for row in result["pairs"]:
            self.assertEqual(row["direct_callees"]["ordinary"], ["same_impl"])
            self.assertEqual(row["direct_callees"]["candidate"], ["same_impl"])
        for kwargs in ({"wrong_mask": True}, {"raw_use": True}):
            result = compare_ir(forwarding_module(**kwargs))
            self.assertTrue(
                all(row["state"] == "unsupported_ir_layout" for row in result["pairs"])
            )
        result = compare_ir(forwarding_module(wrong_callee=True))
        self.assertTrue(all(row["state"] == "different" for row in result["pairs"]))

    def test_separate_module_compares_reachable_helper_bodies(self):
        reference = forwarding_module()
        same = compare_ir(reference, reference_module=reference)
        self.assertEqual(same["state"], "normalized_ir_matches")
        self.assertEqual(same["reference_scope"], "separate_original_module")
        changed = reference.replace(
            "define void @same_impl(<3 x i32> %position) {\n  ret void\n}",
            "define void @same_impl(<3 x i32> %position) {\n  %changed = fadd fast float 1.0, 2.0\n  ret void\n}",
        )
        result = compare_ir(changed, reference_module=reference)
        self.assertNotEqual(result["state"], "normalized_ir_matches")
        self.assertTrue(all(row["normalized_text_equal"] for row in result["pairs"]))
        self.assertTrue(
            all(
                any(h["state"] == "different" for h in row["helper_comparisons"])
                for row in result["pairs"]
            )
        )
        missing = reference[: reference.index("define void @same_impl")]
        self.assertNotEqual(compare_ir(missing)["state"], "normalized_ir_matches")

    def test_unique_filename_initializer_is_compared_not_ignored(self):
        source = forwarding_module()
        original = (
            source
            + "\ndefine internal void @_GLOBAL__sub_I_reference.metal() {\n  ret void\n}"
        )
        candidate = (
            source
            + "\ndefine internal void @_GLOBAL__sub_I_candidate.metal() {\n  ret void\n}"
        )
        result = compare_ir(candidate, reference_module=original)
        self.assertEqual(result["state"], "normalized_ir_matches")
        self.assertEqual(len(result["module_initializer_renames"]), 1)
        changed = candidate.replace(
            "define internal void @_GLOBAL__sub_I_candidate.metal() {\n  ret void",
            "define internal void @_GLOBAL__sub_I_candidate.metal() {\n  %x = fadd float 1.0, 2.0\n  ret void",
        )
        self.assertNotEqual(
            compare_ir(changed, reference_module=original)["state"],
            "normalized_ir_matches",
        )
        ambiguous = (
            candidate
            + "\ndefine internal void @_GLOBAL__sub_I_extra.metal() {\n  ret void\n}"
        )
        self.assertNotEqual(
            compare_ir(ambiguous, reference_module=original)["state"],
            "normalized_ir_matches",
        )

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
