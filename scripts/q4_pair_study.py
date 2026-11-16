#!/usr/bin/env python3
"""Opt-in Q4 pair trained gate and frozen paired development screen."""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.q4_pair import analyze, MARKER
from bench.common import (
    ROOT,
    atomic_json,
    available_memory,
    hardware,
    read_jsonl,
    sha256,
)
from bench.run import run
from scripts.check_kernel_profile import command
from scripts.diagnose_models import build_configuration
from scripts.download_models import download


def pipeline(out, model_dir):
    atomic_json(out / "environment.json", hardware())
    free = available_memory()
    if free < 5 * 2**30:
        raise RuntimeError("Q4 pair study requires at least 5 GiB available")
    python = Path(sys.executable)
    build = ROOT / "build-cpu"
    saved = ROOT / "build-q4-reference"
    saved.mkdir(exist_ok=False)
    for name in ("tokenomics", "CMakeCache.txt"):
        shutil.copy2(build / name, saved / name)
    reference = saved / "tokenomics"
    values = build_configuration(reference)["values"]
    if (
        values["TOKENOMICS_CPU_BATCH_INVARIANT"] != "ON"
        or values["TOKENOMICS_CPU_Q4_PAIR"] != "OFF"
    ):
        raise RuntimeError("Bootstrap must provide the unchanged guarded reference")
    command(
        [
            python.with_name("cmake"),
            "-S",
            ROOT,
            "-B",
            build,
            "-DTOKENOMICS_CPU_Q4_PAIR=ON",
        ],
        120,
    )
    command([python.with_name("cmake"), "--build", build, "-j2"], 1200)
    command(
        [python.with_name("ctest"), "--test-dir", build, "--output-on-failure"], 600
    )
    candidate = build / "tokenomics"
    flags = {
        "guarded": build_configuration(reference),
        "q4-pair": build_configuration(candidate),
    }
    atomic_json(
        out / "build.json",
        {
            "configuration": flags,
            "binary_sha256": {
                "guarded": sha256(reference),
                "q4-pair": sha256(candidate),
            },
            "generated_source_sha256": {
                name: sha256(build / name)
                for name in (
                    "tokenomics-ggml-cpu.c",
                    "tokenomics-x86-quants.c",
                    "tokenomics-tiled.cpp",
                )
            },
        },
    )
    lock = model_dir / "models.lock.json"
    draft = download("qwen25-05", model_dir, lock)
    target = download("qwen25-15", model_dir, lock)
    print("TOKENOMICS_Q4_PAIR_STAGE trained-gate", flush=True)
    gate = out / "q4-gate"
    command(
        [
            python,
            ROOT / "scripts/check_models.py",
            "--binary",
            candidate,
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
            "5400",
        ],
        5500,
    )
    checks = read_jsonl(gate / "correctness.jsonl")
    records = read_jsonl(gate / "candidate.jsonl")
    status = json.loads((gate / "status.json").read_text())
    if (
        len(checks) != 272
        or len(records) != 272
        or not all(c["ok"] for c in checks)
        or status["state"] != "passed"
        or status["synthetic_fixture"] is not False
        or any(MARKER not in r["system_info"] for r in records)
    ):
        raise RuntimeError("Q4 pair failed the unchanged 272-comparison gate")
    if max(r["peak_rss_bytes"] for r in records) > free * 0.8:
        raise RuntimeError("Q4 pair trained gate exceeded memory headroom")
    print("TOKENOMICS_Q4_PAIR_STAGE paired-screen", flush=True)
    cfg = json.loads((ROOT / "configs/q4-pair-screen-v1.json").read_text())
    prompts = [
        p
        for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")
        if p["id"] in cfg["prompt_ids"]
    ]
    if len(prompts) != 3 or any(p["split"] != "development" for p in prompts):
        raise RuntimeError("Frozen development prompts are missing")
    target3 = download("qwen25-3", model_dir, lock)
    exes = {
        c["id"]: candidate if c["kernel_profile"] == "q4-pair" else reference
        for c in cfg["configurations"]
    }
    rows = run(
        reference,
        target3,
        None,
        prompts,
        cfg,
        out / "q4-pair",
        0,
        {
            "synthetic_fixture": False,
            "build_configuration": flags,
            "trained_gate_status_sha256": sha256(gate / "status.json"),
        },
        executables=exes,
    )
    if (
        any(r["peak_rss_bytes"] <= 0 for r in rows)
        or max(r["peak_rss_bytes"] for r in rows) > free * 0.8
    ):
        raise RuntimeError("Q4 pair screen exceeded memory headroom")
    summary = analyze(out / "q4-pair")
    atomic_json(out / "models.lock.json", json.loads(lock.read_text()))
    atomic_json(
        out / "status.json",
        {
            "schema_version": 1,
            "state": "q4_pair_comparison_complete",
            "records": len(rows),
            "trained_comparisons": len(checks),
            "cross_profile_exact": summary["cross_profile_exact"],
            "message": summary["limitation"],
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
