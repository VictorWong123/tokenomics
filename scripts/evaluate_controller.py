#!/usr/bin/env python3
"""Freeze and run the full held-out controller matrix from development priors."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.controller import report
from bench.common import ROOT, atomic_json, read_jsonl, sha256
from bench.controller import configurations, validate_binding
from bench.prompt_fit import check as check_prompt_fit
from bench.run import run, RunPaused


def prepare(prior, prior_sha256, prompts, template, *, synthetic_fixture=False):
    """No evaluation outcomes enter baseline or policy configuration selection."""
    point = prior["provenance"]["operating_point"]
    if (
        prior["provenance"]["split"] != "development"
        or prior["synthetic"] != synthetic_fixture
    ):
        raise ValueError(
            "Evaluation requires correctly labeled frozen development priors"
        )
    heldout = [p for p in prompts if p["split"] == "heldout"]
    if (
        not heldout
        or len({p["id"] for p in heldout}) != len(heldout)
        or {p["id"] for p in heldout} & set(prior["provenance"]["fit_prompt_ids"])
    ):
        raise ValueError("Invalid or overlapping held-out prompts")
    if point["cache"] != "retain":
        raise ValueError("Version 1 evaluation requires retained-cache calibration")
    if not synthetic_fixture:
        counts = {
            category: sum(p["category"] == category for p in heldout)
            for category in ("qa", "code", "summary")
        }
        if len(heldout) != 90 or set(counts.values()) != {30}:
            raise ValueError("Full evaluation requires all 90 frozen held-out prompts")
        if template["output_caps"] != [128, 512] or template["repetitions"] < 5:
            raise ValueError(
                "Full evaluation requires caps128/512 and at least five repeats"
            )
    cfg = copy.deepcopy(template)
    cfg.update(
        phase="evaluation",
        split="heldout",
        detailed=False,
        selected_fixed=prior["provenance"]["selected_fixed"],
        development_selection_sha256=prior_sha256,
        controller_prior_id=prior["prior_id"],
        configurations=configurations(prior),
        **{k: point[k] for k in ("context", "batch", "threads")},
    )
    cfg.setdefault("warmup_prompt", "Warm-up: continue briefly.")
    if point["drafter"] == "model":
        cfg["configurations"].insert(
            2,
            dict(
                id="upstream-style",
                policy="fixed",
                k=4,
                drafter="model",
                cache="upstream",
                detailed=False,
            ),
        )
    return heldout, cfg


def evaluate(
    exe,
    target,
    draft,
    prior_path,
    prompts,
    template,
    out,
    *,
    budget_seconds=6600,
    synthetic_fixture=False,
):
    prior = json.loads(prior_path.read_text())
    selected, cfg = prepare(
        prior,
        sha256(prior_path),
        prompts,
        template,
        synthetic_fixture=synthetic_fixture,
    )
    out.mkdir(parents=True, exist_ok=True)
    point = prior["provenance"]["operating_point"]
    validate_binding(
        prior,
        {
            **point,
            "binary_sha256": sha256(exe),
            "target_sha256": sha256(target),
            "draft_sha256": sha256(draft) if draft else None,
            **{k: cfg[k] for k in ("context", "batch", "threads")},
        },
        synthetic_fixture=synthetic_fixture,
    )
    check_prompt_fit(
        exe, target, selected, cfg, out / "prompt-fit", point["gpu_layers"]
    )
    # run() hashes the executable/models and rejects an incompatible calibration
    # before launching any model work. Changed inputs cannot resume saved blocks.
    try:
        rows = run(
            exe,
            target,
            draft,
            selected,
            cfg,
            out / "evaluation",
            prior["provenance"]["operating_point"]["gpu_layers"],
            {
                "purpose": "frozen-heldout-controller-evaluation",
                "synthetic_fixture": synthetic_fixture,
                "prior_sha256": sha256(prior_path),
            },
            deadline=time.monotonic() + budget_seconds,
        )
    except RunPaused as exc:
        atomic_json(
            out / "status.json", dict(state="paused", message=str(exc), heldout=True)
        )
        return 11
    result = report(out / "evaluation")
    atomic_json(
        out / "status.json",
        dict(
            state="evaluation_complete",
            heldout=True,
            synthetic_fixture=synthetic_fixture,
            records=len(rows),
            prompts=len(selected),
            controller_ratio=result["comparisons"][
                "cost-aware/" + cfg["selected_fixed"]
            ],
        ),
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=ROOT / "build-cpu/tokenomics")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=6600)
    args = parser.parse_args()
    if args.budget_seconds <= 0:
        parser.error("budget must be positive")
    code = evaluate(
        args.exe.resolve(),
        args.target.resolve(),
        args.draft.resolve() if args.draft else None,
        args.prior,
        read_jsonl(ROOT / "bench/prompts-v1.jsonl"),
        json.loads((ROOT / "configs/evaluation-v1.json").read_text()),
        args.out,
        budget_seconds=args.budget_seconds,
    )
    sys.exit(code)
