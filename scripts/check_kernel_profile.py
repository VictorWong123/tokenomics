#!/usr/bin/env python3
"""Fixed trained gate and development screen for the opt-in packed CPU profile."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import (
    ROOT,
    atomic_json,
    hardware,
    available_memory,
    read_jsonl,
    sha256,
    estimate_pilot_budget,
)
from bench.run import run
from scripts.run_all import ngram_config
from scripts.download_models import download
from scripts.diagnose_models import build_configuration
from analysis.timing_screen import report

MARKER = "TOKENOMICS_CPU_PACKED_VERIFY=ON"


def command(args, timeout):
    subprocess.run([str(x) for x in args], cwd=ROOT, check=True, timeout=timeout)


def pipeline(out, model_dir):
    atomic_json(out / "environment.json", hardware())
    python = Path(sys.executable)
    cmake = python.with_name("cmake")
    ctest = python.with_name("ctest")
    build = ROOT / "build-cpu"
    exe = build / "tokenomics"
    print("TOKENOMICS_KERNEL_STAGE configure", flush=True)
    command(
        [
            cmake,
            "-S",
            ROOT,
            "-B",
            build,
            "-DTOKENOMICS_CPU_BATCH_INVARIANT=OFF",
            "-DTOKENOMICS_CPU_PACKED_VERIFY=ON",
        ],
        120,
    )
    command([cmake, "--build", build, "-j2"], 1200)
    command([ctest, "--test-dir", build, "--output-on-failure"], 600)
    settings = build_configuration(exe)
    if settings["values"]["TOKENOMICS_CPU_PACKED_VERIFY"] != "ON":
        raise RuntimeError("Packed profile is not enabled")
    atomic_json(
        out / "kernel-build.json",
        {
            "configuration": settings,
            "binary_sha256": sha256(exe),
            "generated_source_sha256": {
                name: sha256(build / name)
                for name in ("tokenomics-repack.cpp", "tokenomics-tiled.cpp")
            },
        },
    )
    free = available_memory()
    if free < 5 * 2**30:
        raise RuntimeError("Packed profile requires at least 5 GiB available")
    lock = model_dir / "models.lock.json"
    draft = download("qwen25-05", model_dir, lock)
    target = download("qwen25-15", model_dir, lock)
    print("TOKENOMICS_KERNEL_STAGE trained-diagnosis", flush=True)
    command(
        [
            python,
            ROOT / "scripts/diagnose_models.py",
            "--binary",
            exe,
            "--target",
            target,
            "--draft",
            draft,
            "--out",
            out / "kernel-diagnostic",
            "--threads",
            "2",
        ],
        1800,
    )
    diagnosis = json.loads((out / "kernel-diagnostic/status.json").read_text())
    if (
        diagnosis["state"] != "diagnostic_complete"
        or len(diagnosis["full_comparisons"]) != 14
        or not diagnosis["reference_reset_equal"]
        or any(c["mismatch"] is not None for c in diagnosis["full_comparisons"])
    ):
        raise RuntimeError("Packed profile failed exact full-generation diagnosis")
    # Fresh-prefill probes may differ from an incremental history. Preserve
    # their evidence; the full request gate below is the correctness criterion.
    full = read_jsonl(out / "kernel-diagnostic/full.jsonl")
    if len(full) != 14 or any(MARKER not in r["system_info"] for r in full):
        raise RuntimeError("Diagnosis did not execute the packed profile")
    print("TOKENOMICS_KERNEL_STAGE trained-gate", flush=True)
    gate = out / "kernel-gate"
    command(
        [
            python,
            ROOT / "scripts/check_models.py",
            "--binary",
            exe,
            "--target",
            target,
            "--draft",
            draft,
            "--out",
            gate,
            "--threads",
            "2",
            "--context",
            "2048",
            "--batch",
            "512",
            "--max-tokens",
            "32",
            "--timeout-seconds",
            "7200",
        ],
        7800,
    )
    checks = read_jsonl(gate / "correctness.jsonl")
    status = json.loads((gate / "status.json").read_text())
    candidates = read_jsonl(gate / "candidate.jsonl")
    if (
        len(checks) != 272
        or len(candidates) != 272
        or not all(c["ok"] for c in checks)
        or status["state"] != "passed"
        or status["synthetic_fixture"] is not False
        or any(MARKER not in r["system_info"] for r in candidates)
    ):
        raise RuntimeError(
            "Packed profile did not pass the unchanged 272-comparison gate"
        )
    if max(r["peak_rss_bytes"] for r in candidates) > free * 0.8:
        raise RuntimeError("Packed gate failed measured memory headroom")
    print("TOKENOMICS_KERNEL_STAGE development-screen", flush=True)
    target3 = download("qwen25-3", model_dir, lock)
    cfg = ngram_config(json.loads((ROOT / "configs/pilot-v1.json").read_text()))
    cfg["threads"] = 2
    small = {**cfg, "output_caps": [64], "repetitions": 1}
    prompts = read_jsonl(ROOT / "bench/prompts-v1.jsonl")
    chosen = [
        next(
            p
            for p in prompts
            if p["pilot"] and p["category"] == cat and p["length_bucket"] == "long"
        )
        for cat in ("qa", "code", "summary")
    ]
    point = out / "qwen25-05-3-q4-ngram"
    start = time.monotonic()
    rows = run(
        exe,
        target3,
        None,
        chosen,
        small,
        point / "timing",
        0,
        {
            "backend": "cpu",
            "kernel_profile": "packed_verify",
            "synthetic_fixture": False,
            "build_flags": (build / "CMakeCache.txt").read_text(),
            "model_lock": {
                "qwen25-3": json.loads(lock.read_text())["models"]["qwen25-3"]
            },
        },
    )
    if len(rows) != 18 or any(MARKER not in r["system_info"] for r in rows):
        raise RuntimeError("Packed screen identity/count mismatch")
    rss = max(r["peak_rss_bytes"] for r in rows)
    if rss > free * 0.8:
        raise RuntimeError("Packed screen failed measured memory headroom")
    atomic_json(
        point / "duration-estimate.json",
        {
            **estimate_pilot_budget(rows, 18, cfg),
            "sample_wall_seconds": time.monotonic() - start,
        },
    )
    report(point / "timing", point / "analysis")
    atomic_json(out / "models.lock.json", json.loads(lock.read_text()))
    atomic_json(
        out / "status.json",
        {
            "schema_version": 1,
            "state": "kernel_profile_passed",
            "message": "Experimental packed CPU profile passed the trained gate and development screen; no full feasibility/controller result.",
            "checked_comparisons": 272,
            "timing_records": 18,
            "peak_rss_bytes": rss,
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
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise
