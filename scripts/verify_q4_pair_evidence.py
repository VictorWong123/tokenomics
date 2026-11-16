#!/usr/bin/env python3
"""Recheck extracted Q4 pair evidence; archive ownership/shutdown are separate."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.q4_pair import analyze, MARKER
from bench.common import atomic_json, read_jsonl, sha256
from scripts.check_models import check_cycles, compare
from scripts.verify_experiment_evidence import require, verify_grid


def verify_summary(point):
    """Regenerate into a separate directory, never repairing archived evidence."""
    point = Path(point)
    with tempfile.TemporaryDirectory() as tmp:
        recalculated = analyze(point, Path(tmp) / "summary.json")
    retained = json.loads((point / "analysis/summary.json").read_text())
    require(recalculated == retained, "Q4 paired summary differs from raw measurements")
    return recalculated


def check(evidence, source):
    evidence = Path(evidence)
    gate = evidence / "q4-gate"
    status = json.loads((gate / "status.json").read_text())
    manifest = json.loads((gate / "manifest.json").read_text())
    require(
        status["state"] == "passed" and status["synthetic_fixture"] is False,
        "Trained Q4 gate not passed",
    )
    require(
        manifest["source"]["revision"] == source
        and manifest["synthetic_fixture"] is False,
        "Trained Q4 gate source/fixture identity differs",
    )
    refs = read_jsonl(gate / "reference.jsonl") + read_jsonl(
        gate / "boundary-reference.jsonl"
    )
    reference = {r["prompt_id"]: r for r in refs}
    candidates = read_jsonl(gate / "candidate.jsonl")
    checks = read_jsonl(gate / "correctness.jsonl")
    require(
        len(checks) == len(candidates) == 272 and all(c["ok"] for c in checks),
        "Incomplete Q4 trained gate",
    )
    require(
        all(compare(reference[r["prompt_id"]], r) is None for r in candidates),
        "Q4 trained gate output mismatch",
    )
    require(
        all(MARKER in r["system_info"] for r in refs + candidates),
        "Wrong trained Q4 profile",
    )
    check_cycles(candidates, read_jsonl(gate / "candidate.cycles.jsonl"))
    point = evidence / "q4-pair"
    rows, frozen, _ = verify_grid(point, True, source)
    require(
        len(rows) == 72 and frozen["identity"]["synthetic_fixture"] is False,
        "Incomplete trained Q4 paired screen",
    )
    build = json.loads((evidence / "build.json").read_text())
    binaries = build["binary_sha256"]
    require(
        manifest["binary_sha256"] == binaries["q4-pair"]
        and frozen["binary_sha256"] == binaries["guarded"],
        "Q4 build/gate binary changed",
    )
    for conf in frozen["config"]["configurations"]:
        require(
            frozen["configuration_binary_sha256"][conf["id"]]
            == binaries[conf["kernel_profile"]],
            "Paired Q4 executable mapping changed",
        )
    require(
        frozen["identity"]["trained_gate_status_sha256"]
        == sha256(gate / "status.json"),
        "Q4 trained prerequisite changed",
    )
    models = json.loads((evidence / "models.lock.json").read_text())["models"]
    require(
        manifest["target_sha256"] == models["qwen25-15"]["sha256"]
        and manifest["draft_sha256"] == models["qwen25-05"]["sha256"]
        and frozen["target_sha256"] == models["qwen25-3"]["sha256"],
        "Q4 model identity changed",
    )
    recalculated = verify_summary(point)
    return {
        "state": "q4_pair_comparison_complete",
        "exact_comparisons": 272,
        "timing_records": 72,
        "cross_profile_exact": True,
        "binary_sha256": binaries["q4-pair"],
        "reference_binary_sha256": binaries["guarded"],
        "screen_target_sha256": frozen["target_sha256"],
        "peak_rss_bytes": max(r["peak_rss_bytes"] for r in candidates + rows),
        "screen": recalculated,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.out, check(args.evidence, args.source))
