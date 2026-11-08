#!/usr/bin/env python3
"""Bounded confirmation of the near-neutral ngram screen on distinct dev prompts.

This is a targeted additional experiment, not a substitute for full Pilot A/B.
The chosen action and validation grid are frozen before any confirmation data.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import (
    ROOT,
    atomic_json,
    read_jsonl,
    sha256,
    available_memory,
    hardware,
)
from bench.run import run, RunPaused
from analysis.report import validate_run, summarize
from scripts.download_models import download


def analyze(directory):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    cfg = manifest["frozen"]["config"]
    rows = read_jsonl(directory / "requests.jsonl")
    validate_run(rows, manifest, json.loads((directory / "complete.json").read_text()))
    if cfg.get("phase") != "targeted_confirmation" or not cfg.get(
        "fit_analysis_sha256"
    ):
        raise ValueError("Missing frozen confirmation selection")
    result = summarize(rows, frozen_fixed=cfg["selected_fixed"])
    result.pop("q1_supported", None)
    metric = result["comparisons"][cfg["selected_fixed"] + "/target"]
    synthetic = manifest["frozen"].get("identity", {}).get("synthetic_fixture", False)
    result.update(
        phase="targeted_confirmation",
        selected_fixed=cfg["selected_fixed"],
        fit_analysis_sha256=cfg["fit_analysis_sha256"],
        synthetic_fixture=synthetic,
        chosen_fixed_exceeds_practical_margin=not synthetic
        and metric["ratio"] > 1.05
        and metric["ci95"][0] > 1,
        full_feasibility_cleared=False,
        limitation="Frozen-action confirmation on nine distinct development validation prompts. No retuning, full Pilot A/B completion, Q2 decision, controller gain or held-out result follows.",
    )
    atomic_json(directory / "analysis/summary.json", result)
    return result


def pipeline(out, model_dir, budget_seconds=6600):
    atomic_json(out / "environment.json", hardware())
    cfg = json.loads((ROOT / "configs/ngram-confirmation-v1.json").read_text())
    fit_path = ROOT / cfg["fit_manifest"]
    selection = ROOT / cfg["fit_analysis"]
    if (
        sha256(fit_path) != cfg["fit_manifest_sha256"]
        or sha256(selection) != cfg["fit_analysis_sha256"]
    ):
        raise RuntimeError("Frozen fit evidence changed")
    fit = json.loads(fit_path.read_text())["frozen"]
    scores = json.loads(selection.read_text())["geomean_ratios_vs_target"]
    best = max((c for c in scores if c.startswith("fixed")), key=scores.get)
    prompts = [
        p
        for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")
        if p["id"] in cfg["prompt_ids"]
    ]
    if (
        cfg["selected_fixed"] != best
        or len(prompts) != 9
        or any(p["split"] != "development" for p in prompts)
        or {p["id"] for p in prompts} & {p["id"] for p in fit["prompts"]}
    ):
        raise RuntimeError("Frozen selection or independent development split invalid")
    exe = ROOT / "build-cpu/tokenomics"
    if sha256(exe) != fit["binary_sha256"] or any(
        cfg[k] != fit["config"][k]
        for k in ("context", "batch", "threads", "warmup_tokens", "warmup_prompt")
    ):
        raise RuntimeError("Confirmation driver/settings differ from the fit point")
    free = available_memory()
    if free < 5 * 2**30:
        raise RuntimeError("Confirmation requires at least 5 GiB available")
    target = download("qwen25-3", model_dir, model_dir / "models.lock.json")
    if sha256(target) != fit["target_sha256"] or fit["draft_sha256"] is not None:
        raise RuntimeError("Confirmation target/mechanism differs from the fit point")
    directory = out / "confirmation"
    try:
        rows = run(
            exe,
            target,
            None,
            prompts,
            cfg,
            directory,
            0,
            {
                "candidate": "qwen25-3-ngram-confirmation",
                "synthetic_fixture": False,
                "fit_manifest_sha256": cfg["fit_manifest_sha256"],
            },
            deadline=time.monotonic() + budget_seconds,
        )
    except RunPaused as exc:
        atomic_json(
            out / "status.json",
            {"schema_version": 1, "state": "paused", "message": str(exc)},
        )
        return 11
    rss = max(r["peak_rss_bytes"] for r in rows)
    if rss > free * 0.8:
        raise RuntimeError("Measured confirmation RSS fails 20% memory headroom")
    summary = analyze(directory)
    atomic_json(
        out / "status.json",
        {
            "schema_version": 1,
            "state": "confirmation_complete",
            "message": "Frozen-action development confirmation completed; full feasibility and controller remain gated.",
            "records": len(rows),
            "peak_rss_bytes": rss,
            "chosen_fixed_exceeds_practical_margin": summary[
                "chosen_fixed_exceeds_practical_margin"
            ],
        },
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=6600)
    args = parser.parse_args()
    if args.budget_seconds <= 0:
        parser.error("--budget-seconds must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        sys.exit(pipeline(args.run_dir, args.model_dir, args.budget_seconds))
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
