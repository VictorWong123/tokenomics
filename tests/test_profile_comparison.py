import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.profile_comparison import analyze
from bench.common import ROOT, atomic_json, sha256


class ProfileComparisonTests(unittest.TestCase):
    def test_profile_execution_requires_a_frozen_complete_trained_gate(self):
        from scripts.compare_cpu_profiles import pipeline
        cfg = json.loads((ROOT / "configs/cpu-profile-screen-v1.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg["trained_gate"] = "gate.json"
            for overrides in ({"state": "failed"}, {"exact_comparisons": 271},
                              {"shutdown_confirmed": False}):
                atomic_json(root / "gate.json", {
                    "state": "kernel_profile_passed", "exact_comparisons": 272,
                    "shutdown_confirmed": True, **overrides,
                })
                cfg["trained_gate_sha256"] = sha256(root / "gate.json")
                atomic_json(root / "configs/cpu-profile-screen-v1.json", cfg)
                with patch("scripts.compare_cpu_profiles.ROOT", root), patch(
                    "scripts.compare_cpu_profiles.command"
                ) as command, patch("scripts.compare_cpu_profiles.download") as download:
                    with self.assertRaisesRegex(RuntimeError, "complete verified trained"):
                        pipeline(root / "out", root / "models")
                    command.assert_not_called()
                    download.assert_not_called()

    def test_exact_paired_grid_and_execution_identity_are_required(self):
        cfg = json.loads((ROOT / "configs/cpu-profile-screen-v1.json").read_text())
        prompts = [{"id": p, "split": "development"} for p in cfg["prompt_ids"]]
        costs = {"target": 100, "guarded-detailed": 102, "packed-target": 80,
                 "packed-detailed": 81, "guarded-fixed8": 150, "packed-fixed8": 110}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            atomic_json(directory / "run_manifest.json", {"frozen": {
                "config": cfg, "prompts": prompts,
                "configuration_binary_sha256": {c: "a" * 64 for c in costs},
                "identity": {"synthetic_fixture": True},
            }})
            rows = []
            for conf in cfg["configurations"]:
                cid = conf["id"]
                cycles = directory / f"{cid}.cycles.jsonl"
                cycles.write_text(json.dumps({"committed": 2}) + "\n" if conf["detailed"] else "")
                marker = ("TOKENOMICS_CPU_PACKED_VERIFY=ON" if conf["kernel_profile"] == "packed"
                          else "TOKENOMICS_CPU_BATCH_INVARIANT=ON")
                for prompt in prompts:
                    for rep in range(3):
                        rows.append({
                            "prompt_id": prompt["id"], "output_cap": 128, "repetition": rep,
                            "configuration_id": cid, "correctness": "pass",
                            "generated_tokens": 2, "generation_ms": costs[cid],
                            "process_ms": costs[cid] + 10, "peak_rss_bytes": 1024,
                            "system_info": marker,
                            "settings": {
                                **{k: conf[k] for k in ("policy", "k", "drafter", "detailed") if k in conf},
                                **{k: cfg[k] for k in ("context", "batch", "threads")}, "gpu_layers": 0,
                            },
                            "source_cycles": cycles.name, "token_ids": [3, 4],
                            "prompt_ids": [1, 2], "stop": "eos",
                        })
            path = directory / "requests.jsonl"
            def write():
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            write()
            (directory / "correctness.jsonl").write_text((json.dumps({"ok": True}) + "\n") * len(rows))
            atomic_json(directory / "complete.json", {"records": len(rows)})
            result = analyze(directory)
            self.assertAlmostEqual(result["comparisons"]["packed-target/target"]["ratio"], 1.25)
            self.assertAlmostEqual(result["logging_overhead"]["guarded"]["decode_cost_fraction"], 0.02)
            self.assertTrue(result["synthetic_fixture"])
            self.assertFalse(result["full_feasibility_cleared"])
            row = next(r for r in rows if r["configuration_id"] == "packed-target")
            marker = row["system_info"]
            row["system_info"] = "wrong build"
            write()
            with self.assertRaisesRegex(ValueError, "execution differs"):
                analyze(directory)
            row["system_info"] = marker
            row["generated_tokens"] = 3
            write()
            with self.assertRaisesRegex(ValueError, "Generated-token count mismatch"):
                analyze(directory)
            row["generated_tokens"] = 2
            row["token_ids"] = [3, 5]
            write()
            with self.assertRaisesRegex(ValueError, "output mismatch"):
                analyze(directory)
            row["token_ids"] = [3, 4]
            rows.pop()
            write()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analyze(directory)
