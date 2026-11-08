"""Controller configuration and minimal telemetry through the real executable."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def synthetic_prior():
    return {
        "schema_version": 1,
        "prior_id": "synthetic-cli-test-v1",
        "synthetic": True,
        "priors": [
            {"action": a, "committed_tokens": 1, "base_ms": 1 if a == 0 else 100}
            for a in (0, 1, 2, 4, 6, 8)
        ],
        "probe_interval": 2,
        "provenance": {"purpose": "CLI fixture test; no trained-model estimate"},
    }


@unittest.skipUnless(os.getenv("TOKENOMICS_TEST_BINARY"), "requires fixture engine")
class ControllerCLI(unittest.TestCase):
    def test_configuration_telemetry_and_errors(self):
        exe = Path(os.environ["TOKENOMICS_TEST_BINARY"]).resolve()
        model = exe.parent / "fixtures/target.gguf"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            prior = synthetic_prior()
            jobs = [
                {"prompt": "a b", "max_tokens": 19, "policy": "target"},
                *[
                    {
                        "prompt": "a b",
                        "max_tokens": 19,
                        "policy": "cost-aware",
                        "drafter": "ngram",
                        "controller": prior,
                        "detailed": detailed,
                    }
                    for detailed in (True, False)
                ],
            ]
            jobfile = out / "jobs.jsonl"
            jobfile.write_text("".join(json.dumps(j) + "\n" for j in jobs))
            args = [
                str(exe),
                "--target",
                str(model),
                "--threads",
                "1",
                "--context",
                "64",
                "--batch",
                "32",
            ]
            cycles = out / "cycles.jsonl"
            proc = subprocess.run(
                args + ["--jobs", str(jobfile), "--cycles", str(cycles)],
                capture_output=True,
                text=True,
                check=True,
            )
            rows = [json.loads(s) for s in proc.stdout.splitlines()]
            traces = [json.loads(s) for s in cycles.read_text().splitlines()]
            for row in rows[1:]:
                self.assertEqual(row["token_ids"], rows[0]["token_ids"])
                self.assertEqual(row["stop"], rows[0]["stop"])
                self.assertTrue(row["controller_active"])
                self.assertEqual(
                    row["settings"]["controller"]["provenance"], prior["provenance"]
                )
                self.assertTrue(row["settings"]["controller"]["synthetic"])
                self.assertGreaterEqual(row["warm_ms"], row["generation_ms"])
                self.assertGreaterEqual(row["warm_ms"], row["controller_parse_ms"])
                self.assertEqual(
                    sum(a["committed_tokens"] for a in row["execution"]["actions"]), 19
                )
                self.assertGreater(row["controller"]["probes"], 0)
            detailed = [t for t in traces if "controller" in t]
            self.assertEqual(len(detailed), rows[1]["execution"]["cycles"])
            self.assertEqual(sum(t["committed"] for t in detailed), 19)
            self.assertTrue(
                all(len(t["controller"]["candidates"]) == 6 for t in detailed)
            )
            for job in (
                {"policy": "cost-aware", "drafter": "ngram"},
                {"policy": "target", "controller": prior},
                {
                    "policy": "cost-aware",
                    "drafter": "ngram",
                    "controller": {**prior, "alpha": True},
                },
            ):
                jobfile.write_text(json.dumps(job) + "\n")
                failed = subprocess.run(
                    args + ["--jobs", str(jobfile)], capture_output=True, text=True
                )
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn("controller", failed.stderr)
            config = out / "prior.json"
            config.write_text(json.dumps(prior))
            proc = subprocess.run(
                args
                + [
                    "--policy",
                    "cost-aware",
                    "--drafter",
                    "ngram",
                    "--controller",
                    str(config),
                    "--max-tokens",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            row = json.loads(proc.stdout)
            self.assertEqual(row["controller"]["forced_tail_cycles"], 1)
            self.assertEqual(row["controller"]["fallback_cycles"], 0)


if __name__ == "__main__":
    unittest.main()
