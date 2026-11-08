import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from bench.common import ROOT, read_jsonl
from scripts.verify_profile_evidence import check_failure
from scripts.verify_experiment_evidence import retain


class FailedProfileEvidence(unittest.TestCase):
    def test_failed_grid_is_audited_without_performance_estimate_and_rejects_changed_rows(
        self,
    ):
        source = ROOT / "results/verification/codespace-cpu-profiles-20261108/attempt-1"
        expected = "c834026c6bdddc4d219a25945befc318361e74fb"
        result = check_failure(source, expected)
        self.assertEqual(
            result["first_mismatch"], dict(position=36, reference=323, actual=882)
        )
        self.assertEqual(result["within_profile_nonself_checks"], 4)
        self.assertFalse(result["correctness_passed"])
        self.assertFalse(result["performance_summary_produced"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(source, root / "changed")
            path = next(
                (root / "changed").glob(
                    "profiles/blocks/*/attempt-*/packed-detailed.requests.jsonl"
                )
            )
            rows = read_jsonl(path)
            rows[-1]["token_ids"][0] += 1
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "within a profile"):
                check_failure(root / "changed", expected)

    def test_failed_comparison_never_gets_passing_canonical_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "status.json").write_text('{"state":"failed"}\n')
            receipt = dict(
                run_id=42519929357,
                attempt=1,
                source_sha="a" * 40,
                state="failed",
                zip_sha256="b" * 64,
                archive_sha256="c" * 64,
                exact_comparisons=0,
                shutdown_confirmed=True,
                independent_output_checks=True,
                correctness_passed=False,
            )
            path = root / "receipt.json"
            path.write_text(json.dumps(receipt))
            with patch("scripts.verify_experiment_evidence.ROOT", root):
                retain(evidence, path)
            dest = root / "results/verification/codespace-cpu-profiles-20261108"
            self.assertTrue((dest / "failure-verification-attempt-1.json").is_file())
            self.assertFalse((dest / "verification.json").exists())
