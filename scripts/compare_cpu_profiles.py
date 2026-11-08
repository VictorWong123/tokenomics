#!/usr/bin/env python3
"""Fixed paired profile/logging study after the experimental trained gate."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import (
    ROOT,
    atomic_json,
    available_memory,
    hardware,
    read_jsonl,
    sha256,
)
from bench.run import run
from analysis.profile_comparison import analyze
from analysis.controller import verify
from scripts.check_kernel_profile import command
from scripts.diagnose_models import build_configuration
from scripts.download_models import download


def pipeline(out, model_dir):
    cfg = json.loads((ROOT / "configs/cpu-profile-screen-v1.json").read_text())
    proof_path = ROOT / cfg["trained_gate"]
    if sha256(proof_path) != cfg["trained_gate_sha256"]:
        raise RuntimeError("Trained profile gate provenance changed")
    proof = json.loads(proof_path.read_text())
    if (
        proof["state"] != "kernel_profile_passed"
        or proof["exact_comparisons"] != 272
        or not proof["shutdown_confirmed"]
    ):
        raise RuntimeError("A complete verified trained profile gate is required")
    atomic_json(out / "environment.json", hardware())
    free = available_memory()
    if free < 5 * 2**30:
        raise RuntimeError("Profile comparison requires 5 GiB available")
    build = ROOT / "build-cpu"
    guarded_dir = ROOT / "build-profile-guarded"
    guarded_dir.mkdir(exist_ok=False)
    for name in ("tokenomics", "CMakeCache.txt"):
        shutil.copy2(build / name, guarded_dir / name)
    guarded = guarded_dir / "tokenomics"
    guarded_flags = build_configuration(guarded)
    if guarded_flags["values"]["TOKENOMICS_CPU_BATCH_INVARIANT"] != "ON":
        raise RuntimeError("Bootstrap did not produce the guarded reference")
    python = Path(sys.executable)
    command(
        [
            python.with_name("cmake"),
            "-S",
            ROOT,
            "-B",
            build,
            "-DTOKENOMICS_CPU_BATCH_INVARIANT=OFF",
            "-DTOKENOMICS_CPU_PACKED_VERIFY=ON",
        ],
        120,
    )
    command([python.with_name("cmake"), "--build", build, "-j2"], 1200)
    command(
        [python.with_name("ctest"), "--test-dir", build, "--output-on-failure"], 600
    )
    packed = build / "tokenomics"
    if sha256(packed) != proof["binary_sha256"]:
        raise RuntimeError("Packed binary differs from the trained gate")
    lock = model_dir / "models.lock.json"
    target = download("qwen25-3", model_dir, lock)
    if sha256(target) != proof["screen_target_sha256"]:
        raise RuntimeError("Target differs from the trained profile screen")
    prompts = [
        p
        for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")
        if p["id"] in cfg["prompt_ids"]
    ]
    if len(prompts) != 3 or any(p["split"] != "development" for p in prompts):
        raise RuntimeError("Frozen development profile prompts are missing")
    executables = {
        c["id"]: packed if c["kernel_profile"] == "packed" else guarded
        for c in cfg["configurations"]
    }
    rows = run(
        guarded,
        target,
        None,
        prompts,
        cfg,
        out / "profiles",
        0,
        {
            "synthetic_fixture": False,
            "trained_gate_sha256": cfg["trained_gate_sha256"],
            "build_configuration": {
                "guarded": guarded_flags,
                "packed": build_configuration(packed),
            },
        },
        executables=executables,
    )
    rss = max(r["peak_rss_bytes"] for r in rows)
    if any(r["peak_rss_bytes"] <= 0 for r in rows) or rss > free * 0.8:
        raise RuntimeError("Paired profile study failed measured memory headroom")
    verify(out / "profiles")
    analyze(out / "profiles")
    atomic_json(out / "models.lock.json", json.loads(lock.read_text()))
    atomic_json(
        out / "status.json",
        {
            "schema_version": 1,
            "state": "profile_comparison_complete",
            "records": len(rows),
            "peak_rss_bytes": rss,
            "message": "Paired CPU/logging development component study; no full feasibility/controller result.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        pipeline(args.run_dir, args.model_dir)
    except Exception as exc:
        atomic_json(
            args.run_dir / "status.json",
            {
                "schema_version": 1,
                "state": "failed",
                "message": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        raise
