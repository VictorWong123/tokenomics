import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from bench.common import read_jsonl, atomic_json
from scripts.check_models import execute
from scripts.profile_diagnosis import diagnose
from bench.common import ROOT
from scripts.verify_profile_evidence import check_diagnosis


class RetainedProfileDiagnosis(unittest.TestCase):
    source = (
        ROOT / "results/verification/codespace-profile-diagnosis-20261108/attempt-1"
    )
    revision = "45ba7d6aabda5fd4fc0416a8abb0b8440737c6f8"

    def test_real_histories_and_report_failure_are_kept_distinct(self):
        result = check_diagnosis(self.source, self.revision)
        self.assertEqual(result["exact_original_history_reproductions"], 2)
        self.assertFalse(result["performance_summary_produced"])
        self.assertEqual(result["target_probes"]["guarded"][1]["top_id"], 323)
        self.assertEqual(result["target_probes"]["packed"][1]["top_id"], 882)
        receipt = json.loads((self.source.parent / "verification.json").read_text())
        self.assertEqual(receipt["workflow_conclusion"], "failure")
        self.assertEqual(
            receipt["automation_state"], "failed_after_diagnostic_completed"
        )
        self.assertFalse(receipt["report_json_present"])
        self.assertTrue(receipt["shutdown_confirmed"])

    def test_changed_original_job_and_self_reported_top_id_are_rejected(self):
        import shutil

        for variant, message in (("job", "original jobs"), ("probe", "emitted greedy")):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "changed"
                shutil.copytree(self.source, out)
                if variant == "job":
                    path = out / "packed/original-history.jobs.jsonl"
                    rows = read_jsonl(path)
                    rows[-1]["max_tokens"] -= 1
                else:
                    path = out / "packed/original-history.cycles.jsonl"
                    rows = read_jsonl(path)
                    for cycle in rows:
                        for probe in cycle.get("diagnostic_logits", []):
                            if not cycle.get("warmup") and probe["output_index"] == 36:
                                probe["top_id"] = 323
                    summary = json.loads((out / "diagnosis.json").read_text())
                    summary["probes"]["packed"][1]["top_id"] = 323
                    atomic_json(out / "diagnosis.json", summary)
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaisesRegex(ValueError, message):
                    check_diagnosis(out, self.revision)


class ProfileDiagnosis(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("TOKENOMICS_PACKED_TEST_BINARY"), "requires both CPU profiles"
    )
    def test_original_histories_and_logit_rows_are_reproduced_without_performance_claim(
        self,
    ):
        executables = {
            p: Path(os.environ[key]).resolve()
            for p, key in (
                ("guarded", "TOKENOMICS_TEST_BINARY"),
                ("packed", "TOKENOMICS_PACKED_TEST_BINARY"),
            )
        }
        model = executables["guarded"].parent / "fixtures/ngram-cycle.gguf"
        cfg = dict(context=256, batch=32, threads=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            block = root / "source/profiles/blocks/fixture/attempt-0"
            block.mkdir(parents=True)
            atomic_json(
                block / "failure.json",
                dict(scope="Synthetic harness case, not a trained mismatch"),
            )
            for profile, cid in (("guarded", "target"), ("packed", "packed-target")):
                jobs = [
                    dict(
                        prompt="a b",
                        prompt_id="fixture",
                        configuration_id=cid,
                        repetition=0,
                        policy="target",
                        detailed=False,
                        max_tokens=n,
                        output_cap=n,
                        warmup=w,
                    )
                    for n, w in ((2, True), (128, False))
                ]
                a = SimpleNamespace(
                    binary=executables[profile],
                    target=model,
                    draft=None,
                    out=block,
                    **cfg,
                    gpu_layers=0,
                    timeout_seconds=60,
                )
                execute(a, cid, jobs, False)
                # execute writes the named JSONL; retained attempts use *.requests.jsonl.
                (block / f"{cid}.jsonl").rename(block / f"{cid}.requests.jsonl")
            result = diagnose(executables, model, cfg, root / "source", root / "out")
            self.assertTrue(result["both_original_histories_reproduced"])
            self.assertIsNone(result["first_mismatch"])
            self.assertFalse(result["performance_summary_produced"])
            self.assertEqual(
                [p["output_index"] for p in result["probes"]["packed"]], [35, 36]
            )
            path = block / "packed-target.requests.jsonl"
            rows = read_jsonl(path)
            rows[-1]["token_ids"][0] += 1
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(RuntimeError, "does not reproduce"):
                diagnose(executables, model, cfg, root / "source", root / "changed")
