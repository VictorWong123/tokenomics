#!/usr/bin/env python3
"""One entry point; stop explicitly at unavailable access or a scientific gate.

Never silently invent a controller result or run held-out tuning. Reusing a run
directory resumes identical completed blocks; source/binary drift is rejected.
"""
from __future__ import annotations
import argparse
import copy
import json
import pathlib
import subprocess
import sys
import time
import shutil

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench.common import (
    ROOT,
    atomic_json,
    read_jsonl,
    hardware,
    available_memory,
    sha256,
)
from bench.run import run, RunPaused
from bench.replay import investigate
from analysis.report import write_report
from scripts.download_models import download


def save_status(out, state, **fields):
    atomic_json(out / "status.json", {"schema_version": 1, "state": state, **fields})
    print(state + ": " + fields.get("message", ""), flush=True)


def decision_state(decisions):
    """Resource failures are not negative scientific results."""
    tested = [d for d in decisions if "q1_supported" in d]
    if any(d["q1_supported"] and d["q2_supported"] for d in tested):
        return "proceed_controller_implementation", 20
    if any(d["q1_supported"] for d in tested):
        return "narrow_or_investigate", 21
    if tested:
        return "scope_decision_required", 22
    return "resource_blocked", 2


def pipeline(a):
    deadline = (
        time.monotonic() + a.budget_seconds
        if getattr(a, "budget_seconds", None) is not None
        else None
    )
    out = a.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    exe = (ROOT / a.build_dir / "tokenomics").resolve()
    if not exe.exists():
        raise RuntimeError(
            f"Build missing. Run bash scripts/bootstrap.sh {a.backend}, then rerun this command."
        )
    ctest = pathlib.Path(sys.executable).with_name("ctest")
    if not ctest.exists():
        located = shutil.which("ctest")
        if not located:
            raise RuntimeError("ctest missing; run scripts/bootstrap.sh first")
        ctest = pathlib.Path(located)
    subprocess.run(
        [str(ctest), "--test-dir", str(exe.parent), "--output-on-failure"], check=True
    )
    atomic_json(out / "environment.json", hardware())
    cfg = json.loads((ROOT / "configs/pilot-v1.json").read_text())
    if a.threads:
        cfg["threads"] = a.threads
    gpu_layers = 0 if a.backend == "cpu" else 99
    identity = {
        "backend": a.backend,
        "build_flags": (exe.parent / "CMakeCache.txt").read_text(),
        "manifest_sha256": sha256(ROOT / "bench/prompts-v1.jsonl"),
        "synthetic_fixture": a.fixtures_only,
    }
    if a.fixtures_only:
        target = exe.parent / "fixtures/target.gguf"
        draft = target
        cfg.update(
            output_caps=[32, 128],
            repetitions=2,
            context=256,
            batch=32,
            threads=1,
            warmup_tokens=2,
        )
        cfg["configurations"] = [
            x
            for x in cfg["configurations"]
            if x["id"]
            in (
                "target",
                "resident",
                "fixed2",
                "fixed4",
                "confidence",
                "acceptance",
                "upstream-style",
            )
        ]
        prompts = [
            {
                "id": f"development-{cat}-{i:02d}",
                "category": cat,
                "split": "development",
                "prompt": "a b",
                "pilot": True,
            }
            for cat in ("qa", "code", "summary")
            for i in (0, 3)
        ]
        rows = run(
            exe,
            target,
            draft,
            prompts,
            cfg,
            out / "fixture-pipeline",
            gpu_layers,
            identity,
        )
        write_report(out / "fixture-pipeline", out / "fixture-pipeline/analysis")
        investigate(
            exe,
            target,
            draft,
            out / "fixture-pipeline",
            out / "fixture-replay",
            cfg,
            gpu_layers,
        )
        save_status(
            out,
            "portable_checks_passed",
            message="Synthetic transformer correctness and artifact pipeline passed. No real-model feasibility or hardware speedup established.",
            records=len(rows),
        )
        return 0
    catalog = json.loads((ROOT / "configs/models-v1.json").read_text())
    all_prompts = read_jsonl(ROOT / "bench/prompts-v1.jsonl")
    prompts = [p for p in all_prompts if p["pilot"]]
    decisions = []
    lookup_target = None
    for candidate in catalog["candidates"]:
        if a.candidate and candidate["id"] != a.candidate:
            continue
        free = available_memory()
        if free < candidate["min_available_gib"] * 2**30:
            decisions.append(
                {
                    "candidate": candidate["id"],
                    "status": "insufficient_available_memory",
                    "available_bytes": free,
                }
            )
            continue
        model_dir = getattr(a, "model_dir", ROOT / "models")
        lock = model_dir / "models.lock.json"
        draft = download(candidate["draft"], model_dir, lock)
        target = download(candidate["target"], model_dir, lock)
        pair_dir = out / candidate["id"]
        pair_dir.mkdir(exist_ok=True)
        model_lock = json.loads(lock.read_text())
        pair_identity = {
            **identity,
            "model_lock": {
                key: model_lock["models"][key]
                for key in (candidate["target"], candidate["draft"])
            },
            "candidate": candidate["id"],
        }
        # Tokenize every frozen prompt before long execution. No held-out answers
        # are generated; this only checks prompt and requested output fit.
        tokenize_jobs = pair_dir / "tokenize.jobs.jsonl"
        tokenize_jobs.write_text(
            "".join(
                json.dumps(
                    {"prompt_id": p["id"], "prompt": p["prompt"], "max_tokens": 0}
                )
                + "\n"
                for p in all_prompts
            )
        )
        lengths = pair_dir / "prompt-lengths.jsonl"
        if not lengths.exists():
            subprocess.run(
                [
                    str(exe),
                    "--target",
                    str(target),
                    "--jobs",
                    str(tokenize_jobs),
                    "--out",
                    str(lengths),
                    "--context",
                    str(cfg["context"]),
                    "--threads",
                    str(cfg["threads"]),
                    "--gpu-layers",
                    str(gpu_layers),
                ],
                check=True,
            )
        length_rows = read_jsonl(lengths)
        if len(length_rows) != len(all_prompts) or any(
            len(r["prompt_ids"]) + max(cfg["output_caps"]) - 1 > cfg["context"]
            for r in length_rows
        ):
            raise RuntimeError(
                "Frozen prompt/output caps exceed context or tokenization incomplete. Change manifest before any evaluation, use new run directory."
            )
        # Time a small balanced batch before committing to the full pilot.
        small = copy.deepcopy(cfg)
        small.update(output_caps=[64], repetitions=1)
        small["configurations"] = [
            x
            for x in cfg["configurations"]
            if x["id"] in ("target", "fixed4", "confidence")
        ]
        timing_prompts = [
            next(
                p
                for p in prompts
                if p["category"] == cat and p["length_bucket"] == "long"
            )
            for cat in ("qa", "code", "summary")
        ]
        start = time.monotonic()
        sample = run(
            exe,
            target,
            draft,
            timing_prompts,
            small,
            pair_dir / "timing",
            gpu_layers,
            pair_identity,
            deadline=deadline,
        )
        elapsed = time.monotonic() - start
        measured_execution = sum(r["process_ms"] for r in sample) / 1000
        # Process time includes load, excluded warm-up and measured execution.
        # Scale generated budgets conservatively; EOS makes the estimate rough.
        estimated = (
            measured_execution
            * (len(prompts) / len(timing_prompts))
            * cfg["repetitions"]
            * (sum(cfg["output_caps"]) / 64)
            * (len(cfg["configurations"]) / len(small["configurations"]))
        )
        atomic_json(
            pair_dir / "duration-estimate.json",
            {
                "sample_wall_seconds": elapsed,
                "sample_execution_seconds": measured_execution,
                "sample_prompt_ids": [p["id"] for p in timing_prompts],
                "estimated_full_pilot_seconds": estimated,
                "assumption": "rough linear token/config scaling incl load/warm-up; EOS/prefill/cache/shared CPU can change it in either direction",
            },
        )
        print(
            f'Estimated complete Pilot A: {estimated/3600:.2f} hours for {candidate["id"]}. Results persist after each configuration; identical reruns resume complete blocks.',
            flush=True,
        )
        rss = max(r["peak_rss_bytes"] for r in sample)
        if rss > free * 0.8:
            decisions.append(
                {
                    "candidate": candidate["id"],
                    "status": "memory_headroom_failed",
                    "peak_rss_bytes": rss,
                }
            )
            continue
        lookup_target = target
        if a.timing_only:
            save_status(
                out,
                "timing_complete",
                message="Development timing screen completed. Full Pilot A/B and feasibility decisions were not run.",
                candidate=candidate["id"],
                estimated_full_pilot_seconds=estimated,
                peak_rss_bytes=rss,
            )
            return 10
        if a.max_hours is not None and estimated > a.max_hours * 3600:
            save_status(
                out,
                "duration_limit",
                message=f"Estimate exceeds --max-hours={a.max_hours}; rerun with a larger explicit budget to resume.",
                candidate=candidate["id"],
            )
            return 12
        rows = run(
            exe,
            target,
            draft,
            prompts,
            cfg,
            pair_dir / "pilot-a",
            gpu_layers,
            pair_identity,
            deadline=deadline,
        )
        summary = write_report(pair_dir / "pilot-a", pair_dir / "pilot-a/analysis")
        q2 = investigate(
            exe,
            target,
            draft,
            pair_dir / "pilot-a",
            pair_dir / "pilot-b",
            cfg,
            gpu_layers,
            deadline=deadline,
        )
        decision = {
            "candidate": candidate["id"],
            "q1_supported": summary["q1_supported"],
            "q2_supported": q2["q2_supported"],
            "pilot_a": str(pair_dir / "pilot-a"),
            "pilot_b": str(pair_dir / "pilot-b"),
        }
        decisions.append(decision)
        atomic_json(out / "feasibility.json", decisions)
        if summary["q1_supported"] and q2["q2_supported"]:
            save_status(
                out,
                "proceed_controller_implementation",
                message="Both development feasibility gates passed. Freeze these artifacts for controller implementation; held-out evaluation remains locked. This is not a completed adaptive project.",
                decisions=decisions,
            )
            return 20
    if not decisions:
        raise RuntimeError("unknown or unavailable candidate")
    state, code = decision_state(decisions)
    if code == 2:
        save_status(
            out,
            "resource_blocked",
            message="No candidate passed the resource screen; neither feasibility premise was tested.",
            decisions=decisions,
        )
        return 2
    if code == 21:
        save_status(
            out,
            state,
            message="Independent-model candidates investigated: speculation helps, but cycle-level prediction is unresolved. Inspect Pilot B before choosing a narrower contribution.",
            decisions=decisions,
        )
        return code
    # One cheaper supported drafting mechanism after independent model candidates.
    if lookup_target is not None and not a.skip_ngram:
        target = lookup_target
        ng = copy.deepcopy(cfg)
        ng["configurations"] = [{"id": "target", "policy": "target"}] + [
            {"id": f"fixed{k}", "policy": "fixed", "k": k, "drafter": "ngram"}
            for k in (1, 2, 4, 6, 8)
        ]
        ngdir = out / "ngram-pilot"
        run(
            exe,
            target,
            None,
            prompts,
            ng,
            ngdir,
            gpu_layers,
            {**identity, "candidate": "ngram-simple", "target_sha256": sha256(target)},
            deadline=deadline,
        )
        ns = write_report(ngdir, ngdir / "analysis")
        nq = investigate(
            exe,
            target,
            None,
            ngdir,
            out / "ngram-replay",
            ng,
            gpu_layers,
            "ngram",
            deadline=deadline,
        )
        decisions.append(
            {
                "candidate": "ngram-simple",
                "q1_supported": ns["q1_supported"],
                "q2_supported": nq["q2_supported"],
            }
        )
    atomic_json(out / "feasibility.json", decisions)
    state, code = decision_state(decisions)
    save_status(
        out,
        state,
        message="Bounded investigation finished; inspect the separate Q1/Q2 records for the selected direction. Controller implementation and held-out evaluation remain locked until justified; no silent pivot.",
        decisions=decisions,
    )
    return code


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["cpu", "metal", "cuda"], default="cpu")
    p.add_argument("--build-dir", type=pathlib.Path)
    p.add_argument("--run-dir", type=pathlib.Path, required=True)
    p.add_argument("--fixtures-only", action="store_true")
    p.add_argument("--threads", type=int)
    p.add_argument("--candidate")
    p.add_argument(
        "--model-dir",
        type=pathlib.Path,
        default=ROOT / "models",
        help="Verified model cache and models.lock.json directory",
    )
    p.add_argument("--max-hours", type=float)
    p.add_argument(
        "--budget-seconds",
        type=float,
        help="Pause full Pilot A at a complete block boundary; resume unchanged (exit 11)",
    )
    p.add_argument(
        "--timing-only",
        action="store_true",
        help="Stop after the first resource-safe candidate timing screen (exit 10); no full pilot",
    )
    p.add_argument("--skip-ngram", action="store_true")
    a = p.parse_args()
    if a.timing_only and a.fixtures_only:
        p.error("--timing-only and --fixtures-only are mutually exclusive")
    if a.build_dir is None:
        a.build_dir = pathlib.Path("build-" + a.backend)
    try:
        sys.exit(pipeline(a))
    except RunPaused as ex:
        save_status(a.run_dir, "paused", message=str(ex))
        sys.exit(11)
    except KeyboardInterrupt:
        save_status(
            a.run_dir,
            "interrupted",
            message="Rerun the identical command; completed blocks are retained.",
        )
        sys.exit(130)
    except Exception as ex:
        save_status(a.run_dir, "blocked", message=str(ex))
        print(str(ex), file=sys.stderr)
        sys.exit(2)
