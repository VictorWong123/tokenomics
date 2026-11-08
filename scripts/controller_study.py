#!/usr/bin/env python3
"""Controller-focused trained correctness, calibration and bounded measurement."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import (
    ROOT,
    atomic_json,
    available_memory,
    hardware,
    read_jsonl,
    sha256,
)
from bench.controller import calibrate, configurations
from bench.replay import investigate, run_jobs
from bench.run import run, RunPaused
from analysis.controller import report, plot_traces
from scripts.download_models import download
from scripts.diagnose_models import build_configuration
from bench.cpu_profile import (
    study_configuration,
    profile_matches,
    verified_packed_gate,
)


def correctness_gate(exe, target, draft, prompts, cfg, out, drafter):
    """Explicit synthetic priors exercise paths; these times are not benchmarks."""
    prior = dict(
        schema_version=1,
        prior_id="synthetic-trained-correctness-only-v1",
        synthetic=True,
        probe_interval=2,
        priors=[
            dict(action=a, committed_tokens=1, base_ms=1e-9 if a == 0 else 1e9)
            for a in (0, 1, 2, 4, 6, 8)
        ],
    )
    jobs = []
    prompt = prompts[0]
    for cap in (1, 9, 32):
        meta = dict(
            run_id="controller-correctness",
            prompt_id=prompt["id"],
            repetition=cap,
            warmup=False,
            prompt=prompt["prompt"],
            max_tokens=cap,
            drafter=drafter,
        )
        jobs.append({**meta, "configuration_id": "target", "policy": "target"})
        for cache in ("retain", "upstream"):
            for variant, override in (
                ("cost-aware", {}),
                ("no-fallback", {"allow_target_only": False}),
                ("no-probes", {"enable_probes": False}),
            ):
                jobs.append(
                    {
                        **meta,
                        "configuration_id": cache + "-" + variant,
                        "policy": "cost-aware",
                        "cache": cache,
                        "controller": {**copy.deepcopy(prior), **override},
                    }
                )
    jobs += [
        {**j, "configuration_id": j["configuration_id"] + "-reset", "detailed": False}
        for j in jobs
        if j["max_tokens"] == 32
    ]
    rows, cycles = run_jobs(exe, target, draft, jobs, out, cfg, 0)
    references = {r["repetition"]: r for r in rows if r["configuration_id"] == "target"}
    for row in rows:
        ref = references[row["repetition"]]
        if any(row[k] != ref[k] for k in ("token_ids", "prompt_ids", "stop")):
            raise RuntimeError("Trained controller correctness divergence")
        if row["generated_tokens"] != len(row["token_ids"]):
            raise RuntimeError("Trained controller count divergence")
        if row["settings"]["policy"] == "cost-aware" and not row["controller_active"]:
            raise RuntimeError("Trained controller did not execute")
    status = dict(
        state="passed",
        correctness_only=True,
        synthetic_priors=True,
        drafter=drafter,
        records=len(rows),
        comparisons=len(rows),
        actions_observed=sorted({c["action"] for c in cycles}),
        synced_tokens=sum(c["synced"] for c in cycles),
        binary_sha256=sha256(exe),
        target_sha256=sha256(target),
        draft_sha256=sha256(draft) if draft else None,
    )
    atomic_json(out / "status.json", status)
    return status


def execute_study(
    exe, target, draft, prompts, cfg, out, *, synthetic_fixture=False, deadline=None
):
    """Pure local orchestration, also exercised using generated transformer fixtures."""
    out.mkdir(parents=True, exist_ok=True)
    fit = [p for p in prompts if p["id"] in cfg["fit_prompt_ids"]]
    validation = [p for p in prompts if p["id"] in cfg["validation_prompt_ids"]]
    if (
        len(fit) != len(cfg["fit_prompt_ids"])
        or len(validation) != len(cfg["validation_prompt_ids"])
        or set(cfg["fit_prompt_ids"]) & set(cfg["validation_prompt_ids"])
        or any(p["split"] != "development" for p in fit + validation)
    ):
        raise ValueError("Missing or overlapping development fit/validation prompts")
    drafter = cfg["drafter"]
    identity = {"purpose": cfg["phase"], "synthetic_fixture": synthetic_fixture}
    fit_cfg = {
        **cfg,
        "phase": "controller_fit",
        "output_caps": cfg["fit_output_caps"],
        "repetitions": cfg["fit_repetitions"],
        "detailed": False,
        "configurations": [{"id": "target", "policy": "target"}]
        + [
            {"id": f"fixed{a}", "policy": "fixed", "k": a, "drafter": drafter}
            for a in (1, 2, 4, 6, 8)
        ],
    }
    print("TOKENOMICS_CONTROLLER_STAGE fit", flush=True)
    run(
        exe,
        target,
        draft,
        fit,
        fit_cfg,
        out / "fit",
        identity=identity,
        deadline=deadline,
    )
    print("TOKENOMICS_CONTROLLER_STAGE original-state-calibration", flush=True)
    investigate(
        exe,
        target,
        draft,
        out / "fit",
        out / "replay",
        fit_cfg,
        drafter=drafter,
        deadline=deadline,
    )
    prior = calibrate(
        out / "fit", out / "replay", drafter, synthetic_fixture=synthetic_fixture
    )
    if (out / "prior.json").exists() and json.loads(
        (out / "prior.json").read_text()
    ) != prior:
        raise ValueError("Frozen controller prior changed on resume")
    atomic_json(out / "prior.json", prior)
    validation_cfg = {
        **cfg,
        "selected_fixed": prior["provenance"]["selected_fixed"],
        "development_selection_sha256": sha256(out / "prior.json"),
        "configurations": configurations(prior),
    }
    print("TOKENOMICS_CONTROLLER_STAGE validation", flush=True)
    rows = run(
        exe,
        target,
        draft,
        validation,
        validation_cfg,
        out / "validation",
        identity=identity,
        deadline=deadline,
    )
    result = report(out / "validation")
    # Fresh trace requests and a paired logging comparison are a separate grid.
    # Selection uses the declared median/worst rule; no policy tuning follows.
    selected = set(result["trace_prompt_selection"].values())
    trace_prompts = [p for p in validation if p["id"] in selected]
    trace_cfg = {
        **validation_cfg,
        "phase": "controller_logging_diagnostic",
        "output_caps": [max(cfg["output_caps"])],
        "repetitions": 2,
        "configurations": [
            {"id": "target", "policy": "target", "detailed": False},
            {
                "id": validation_cfg["selected_fixed"],
                "policy": "fixed",
                "drafter": drafter,
                "k": int(validation_cfg["selected_fixed"].removeprefix("fixed")),
                "detailed": False,
            },
            *[
                dict(
                    id=name,
                    policy="cost-aware",
                    drafter=drafter,
                    detailed=detailed,
                    controller=copy.deepcopy(prior),
                )
                for name, detailed in (
                    ("cost-aware", False),
                    ("cost-aware-detailed", True),
                )
            ],
        ],
    }
    print("TOKENOMICS_CONTROLLER_STAGE logging-and-traces", flush=True)
    run(
        exe,
        target,
        draft,
        trace_prompts,
        trace_cfg,
        out / "logging",
        identity=identity,
        deadline=deadline,
    )
    logging = report(out / "logging")
    plot_traces(out / "logging")
    return rows, result, logging


def pipeline(out, model_dir, budget_seconds=6600, *, cpu_profile="guarded"):
    cfg = study_configuration(ROOT, cpu_profile)
    proof = verified_packed_gate(ROOT, cfg) if cpu_profile == "packed" else None
    atomic_json(out / "environment.json", hardware())
    exe = ROOT / "build-cpu/tokenomics"
    if proof is not None:
        from scripts.check_kernel_profile import command

        python = Path(sys.executable)
        command(
            [
                python.with_name("cmake"),
                "-S",
                ROOT,
                "-B",
                ROOT / "build-cpu",
                "-DTOKENOMICS_CPU_BATCH_INVARIANT=OFF",
                "-DTOKENOMICS_CPU_PACKED_VERIFY=ON",
            ],
            120,
        )
        command([python.with_name("cmake"), "--build", ROOT / "build-cpu", "-j2"], 1200)
        command(
            [
                python.with_name("ctest"),
                "--test-dir",
                ROOT / "build-cpu",
                "--output-on-failure",
            ],
            600,
        )
        if sha256(exe) != proof["binary_sha256"]:
            raise RuntimeError(
                "Packed controller executable differs from the trained gate"
            )
    build = build_configuration(exe)
    if not profile_matches(build, cpu_profile):
        raise RuntimeError(
            "Controller study execution profile differs from its protocol"
        )
    atomic_json(
        out / "controller-build.json",
        {"configuration": build, "binary_sha256": sha256(exe)},
    )
    prompts = read_jsonl(ROOT / "bench/prompts-v1.jsonl")
    free = available_memory()
    if free < 5 * 2**30:
        raise RuntimeError("Controller study requires at least 5 GiB available")
    lock = model_dir / "models.lock.json"
    small_target = download("qwen25-15", model_dir, lock)
    draft = download("qwen25-05", model_dir, lock)
    target = download(cfg["target_model"], model_dir, lock)
    if proof is not None and sha256(target) != proof["screen_target_sha256"]:
        raise RuntimeError("Packed controller target differs from the trained gate")
    print("TOKENOMICS_CONTROLLER_STAGE trained-correctness", flush=True)
    fit_prompts = [p for p in prompts if p["id"] in cfg["fit_prompt_ids"]]
    gates = [
        correctness_gate(
            exe, small_target, draft, fit_prompts, cfg, out / "gate-model", "model"
        ),
        correctness_gate(
            exe, target, None, fit_prompts, cfg, out / "gate-ngram", "ngram"
        ),
    ]
    try:
        rows, result, logging = execute_study(
            exe,
            target,
            None,
            prompts,
            cfg,
            out,
            deadline=time.monotonic() + budget_seconds,
        )
    except RunPaused as exc:
        atomic_json(
            out / "status.json",
            dict(schema_version=1, state="paused", message=str(exc), gates=gates),
        )
        return 11
    peak = max(r["peak_rss_bytes"] for r in rows)
    if peak > free * 0.8:
        raise RuntimeError("Controller study failed 20% memory headroom")
    atomic_json(
        out / "status.json",
        dict(
            schema_version=1,
            state="controller_study_complete",
            records=len(rows),
            peak_rss_bytes=peak,
            gates=gates,
            controller_ratio=result["comparisons"]["cost-aware/target"],
            logging_ratio=logging["comparisons"]["cost-aware-detailed/cost-aware"],
            full_feasibility_cleared=False,
            heldout=False,
        ),
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=6600)
    parser.add_argument(
        "--cpu-profile", choices=("guarded", "packed"), default="guarded"
    )
    args = parser.parse_args()
    if args.budget_seconds <= 0:
        parser.error("budget must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        sys.exit(
            pipeline(
                args.run_dir,
                args.model_dir,
                args.budget_seconds,
                cpu_profile=args.cpu_profile,
            )
        )
    except Exception as exc:
        atomic_json(
            args.run_dir / "status.json",
            dict(state="failed", error_type=type(exc).__name__, message=str(exc)),
        )
        raise
