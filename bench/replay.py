"""Pilot B: common-prefix action replay, fit/validation on different prompts.

Setup/prefill is diagnostic and excluded from cycle action cost; all work inside
the cycle remains charged. Transition tests are timed as actual generation.
"""

from __future__ import annotations
import json
import pathlib
import random
import statistics
import subprocess
import math
import time
import numpy as np
from bench.common import atomic_json, read_jsonl, sha256, source_fingerprint
from bench.run import first_mismatch, RunPaused


def run_jobs(exe, target, draft, jobs, directory, cfg, gpu_layers):
    directory.mkdir(parents=True, exist_ok=True)
    jobpath = directory / "jobs.jsonl"
    encoded = "".join(json.dumps(j) + "\n" for j in jobs)
    if jobpath.exists() and jobpath.read_text() != encoded:
        raise RuntimeError("replay job changed; use new directory")
    jobpath.write_text(encoded)
    done = directory / "complete.json"
    if done.exists():
        complete = json.loads(done.read_text())
        for name in ("requests.jsonl", "cycles.jsonl"):
            if complete.get("sha256", {}).get(name) != sha256(directory / name):
                raise RuntimeError(
                    "completed replay artifact changed or predates integrity checks"
                )
        return read_jsonl(directory / "requests.jsonl"), read_jsonl(
            directory / "cycles.jsonl"
        )
    attempt = len(list(directory.glob("attempt-*")))
    d = directory / f"attempt-{attempt}"
    d.mkdir()
    args = [
        str(exe),
        "--target",
        str(target),
        "--jobs",
        str(jobpath),
        "--out",
        str(d / "requests.jsonl"),
        "--cycles",
        str(d / "cycles.jsonl"),
        "--context",
        str(cfg["context"]),
        "--batch",
        str(cfg["batch"]),
        "--threads",
        str(cfg["threads"]),
        "--gpu-layers",
        str(gpu_layers),
    ]
    if draft:
        args += ["--draft", str(draft)]
    # Excluded warm-up primes weights/graphs before any action measurement.
    warm = {**jobs[0], "warmup": True, "max_tokens": 9, "max_cycles": 1}
    attempt_jobs = d / "jobs.jsonl"
    attempt_jobs.write_text(json.dumps(warm) + "\n" + encoded)
    args[args.index("--jobs") + 1] = str(attempt_jobs)
    with open(d / "stderr.txt", "w") as f:
        subprocess.run(args, stderr=f, check=True)
    rows = read_jsonl(d / "requests.jsonl")
    cycles = read_jsonl(d / "cycles.jsonl")
    rows = [r for r in rows if not r.get("warmup", False)]
    cycles = [c for c in cycles if not c.get("warmup", False)]
    if len(rows) != len(jobs):
        raise RuntimeError("incomplete replay records")
    for name, values in (("requests.jsonl", rows), ("cycles.jsonl", cycles)):
        (directory / name).write_text("".join(json.dumps(x) + "\n" for x in values))
    atomic_json(
        done,
        {
            "jobs": len(jobs),
            "attempt": attempt,
            "sha256": {
                n: sha256(directory / n) for n in ("requests.jsonl", "cycles.jsonl")
            },
        },
    )
    return rows, cycles


def investigate(
    exe,
    target,
    draft,
    pilot_dir,
    out,
    cfg,
    gpu_layers=0,
    drafter="model",
    *,
    deadline=None,
):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((pathlib.Path(pilot_dir) / "run_manifest.json").read_text())
    if not (pathlib.Path(pilot_dir) / "complete.json").exists():
        raise RuntimeError("Pilot A must be complete before checkpoint replay")
    operating_point = manifest["frozen"]
    if (
        sha256(exe) != operating_point["binary_sha256"]
        or sha256(target) != operating_point["target_sha256"]
        or (sha256(draft) if draft else None) != operating_point["draft_sha256"]
        or gpu_layers != operating_point["gpu_layers"]
        or any(
            cfg[k] != operating_point["config"][k]
            for k in ("context", "batch", "threads")
        )
    ):
        raise RuntimeError(
            "Replay binary, models or execution settings differ from Pilot A"
        )
    frozen = {
        "code": source_fingerprint(),
        "binary": sha256(exe),
        "target": sha256(target),
        "draft": sha256(draft) if draft else None,
        "config": cfg,
        "gpu_layers": gpu_layers,
        "drafter": drafter,
        "pilot": sha256(pathlib.Path(pilot_dir) / "run_manifest.json"),
    }
    if (out / "manifest.json").exists() and json.loads(
        (out / "manifest.json").read_text()
    ) != frozen:
        raise RuntimeError("replay resume fingerprint changed")
    atomic_json(out / "manifest.json", frozen)
    rows = read_jsonl(pathlib.Path(pilot_dir) / "requests.jsonl")
    refs = {}
    for r in rows:
        if (
            r["configuration_id"] == "target"
            and r["output_cap"] == max(cfg["output_caps"])
            and r["repetition"] == 0
        ):
            refs[r["prompt_id"]] = r
    actions = [0, 1, 2, 4, 6, 8]
    replay_actions = actions + ["dynamic"]
    rng = random.Random(cfg["seed"] + 1)
    measurements = []
    transitions = []
    longest_group = 0.0

    def replay(jobs, directory):
        nonlocal longest_group
        if (
            deadline is not None
            and not (directory / "complete.json").exists()
            and time.monotonic() + 1.25 * longest_group >= deadline
        ):
            raise RunPaused(
                "Pilot B paused between checkpoint groups; resume unchanged"
            )
        start = time.monotonic()
        result = run_jobs(exe, target, draft, jobs, directory, cfg, gpu_layers)
        longest_group = max(longest_group, time.monotonic() - start)
        return result

    for pid, ref in sorted(refs.items()):
        tokens = ref["token_ids"]
        prompt_ids = ref["prompt_ids"]
        jobs = []
        # No checkpoint after the final target trace token, which might be EOG.
        checkpoints = [x for x in (32, 96, 224, 400) if x + 9 < len(tokens)]
        for cp in checkpoints:
            for rep in range(3):
                order = list(replay_actions)
                rng.shuffle(order)
                for action in order:
                    jobs.append(
                        {
                            "run_id": out.name,
                            "configuration_id": f"action{action}",
                            "prompt_id": pid,
                            "repetition": rep,
                            "checkpoint": cp,
                            "policy": (
                                "confidence"
                                if action == "dynamic" and drafter == "model"
                                else "target" if action == 0 else "fixed"
                            ),
                            "k": 8 if action == "dynamic" else action or 4,
                            "drafter": drafter,
                            "prefix_tokens": prompt_ids + tokens[:cp],
                            "max_tokens": 9,
                            "max_cycles": 1,
                            "warmup": False,
                        }
                    )
            # Available historical signal: completed canonical k=4 probe at cp-8.
            jobs.append(
                {
                    "run_id": out.name,
                    "configuration_id": "signal",
                    "prompt_id": pid,
                    "repetition": 0,
                    "checkpoint": cp,
                    "policy": "fixed",
                    "k": 4,
                    "drafter": drafter,
                    "prefix_tokens": prompt_ids + tokens[: cp - 8],
                    "max_tokens": 9,
                    "max_cycles": 1,
                }
            )
        if not jobs:
            continue
        rr, cc = [], []
        for cp in checkpoints:
            r, c = replay(
                [j for j in jobs if j["checkpoint"] == cp],
                out / pid / f"checkpoint-{cp}",
            )
            rr.extend(r)
            cc.extend(c)
        for r in rr:
            cp = r["checkpoint"] - (8 if r["configuration_id"] == "signal" else 0)
            expect = tokens[cp : cp + len(r["token_ids"])]
            mismatch = first_mismatch(expect, r["token_ids"])
            if mismatch is not None:
                atomic_json(
                    out / "correctness_failure.json",
                    {"prompt_id": pid, "checkpoint": cp, "first_mismatch": mismatch},
                )
                raise RuntimeError("replay token mismatch")
        for cp in checkpoints:
            signal = [
                c
                for c in cc
                if c["configuration_id"] == "signal" and c["checkpoint"] == cp
            ][0]
            outcomes = {}
            for action in replay_actions:
                cs = [
                    c
                    for c in cc
                    if c["configuration_id"] == f"action{action}"
                    and c["checkpoint"] == cp
                ]
                outcomes[str(action)] = {
                    "rate": statistics.median(
                        c["committed"] / c["wall_ms"] for c in cs
                    ),
                    "ms": statistics.median(c["wall_ms"] for c in cs),
                    "tokens": statistics.median(c["committed"] for c in cs),
                }
            measurements.append(
                {
                    "prompt_id": pid,
                    "checkpoint": cp,
                    "acceptance": signal["accepted"] / max(signal["actual_drafts"], 1),
                    "context": len(prompt_ids) + cp,
                    "probe_ms": signal["wall_ms"],
                    "outcomes": outcomes,
                }
            )
        if drafter == "model":
            job = {
                "run_id": out.name,
                "configuration_id": "transition",
                "prompt_id": pid,
                "repetition": 0,
                "policy": "script",
                "script": [0] * 16 + [4],
                "prompt": next(
                    p["prompt"] for p in manifest["frozen"]["prompts"] if p["id"] == pid
                ),
                "max_tokens": 64,
                "k": 4,
            }
            tr, tc = replay([job], out / pid / "transition")
            mismatch = first_mismatch(
                tokens[: len(tr[0]["token_ids"])], tr[0]["token_ids"]
            )
            if mismatch is not None:
                raise RuntimeError("transition token mismatch")
            transitions += [
                c["sync_ms"] for c in tc if c["action"] > 0 and c["stale_before"] >= 16
            ]
    atomic_json(out / "checkpoints.json", measurements)
    answer = fit_validate(
        measurements, statistics.median(transitions) if transitions else 0, cfg["seed"]
    )
    if manifest["frozen"].get("identity", {}).get("synthetic_fixture", False):
        answer["synthetic_fixture"] = True
        answer["q2_supported"] = False
        answer["status"] = (
            "Synthetic transformer pipeline check; not scheduling feasibility evidence"
        )
    answer["transition_sync_ms"] = transitions
    answer["dynamic_baseline"] = (
        "pinned upstream confidence, cap 8"
        if drafter == "model"
        else "upstream variable-length ngram lookup, cap 8 (overlaps fixed8)"
    )
    atomic_json(out / "decision.json", answer)
    return answer


def fit_validate(data, transition_ms, seed=20261107):
    fit = [r for r in data if int(r["prompt_id"].rsplit("-", 1)[1]) % 6 < 3]
    val = [r for r in data if int(r["prompt_id"].rsplit("-", 1)[1]) % 6 >= 3]
    if not fit or not val:
        return {
            "schema_version": 2,
            "q2_supported": False,
            "status": "insufficient common-prefix checkpoints",
        }
    if any("dynamic" not in r["outcomes"] for r in data):
        return {
            "schema_version": 2,
            "q2_supported": False,
            "status": "missing existing dynamic baseline; replay required",
        }
    actions = [a for a in fit[0]["outcomes"] if a.isdigit()]

    def bucket(r):
        return (r["acceptance"] >= 0.5, r["context"] >= 256)

    def best(rows):
        return max(
            actions,
            key=lambda a: statistics.mean(
                math.log(r["outcomes"][a]["rate"]) for r in rows
            ),
        )

    global_action = best(fit)
    rules = {}
    for r in fit:
        b = bucket(r)
        if b not in rules:
            rules[b] = best([x for x in fit if bucket(x) == b])
    by_prompt = {}
    dynamic_by_prompt = {}
    winners = set()
    trajectories = {}
    for r in val:
        action = rules.get(bucket(r), global_action)
        o = r["outcomes"][action]
        g = r["outcomes"][global_action]
        # Conservative feasibility estimate: full canonical probe every 16 cycles,
        # fixed 0.05 ms decision allowance, transition catch-up over a 4-cycle hold.
        # It is a declared estimate, not a measured deployable controller gain.
        charged = (
            o["ms"]
            + 0.05
            + r["probe_ms"] / 16
            + (transition_ms / 4 if action != "0" else 0)
        )
        ratio = (o["tokens"] / charged) / g["rate"]
        by_prompt.setdefault(r["prompt_id"], []).append(math.log(ratio))
        dynamic_ratio = (o["tokens"] / charged) / r["outcomes"]["dynamic"]["rate"]
        dynamic_by_prompt.setdefault(r["prompt_id"], []).append(math.log(dynamic_ratio))
        winners.add(max(actions, key=lambda a: r["outcomes"][a]["rate"]))
        trajectories.setdefault(r["prompt_id"], set()).add(action)
    values = np.array([statistics.mean(v) for v in by_prompt.values()])
    rng = np.random.default_rng(seed)
    draws = np.exp(rng.choice(values, (10000, len(values)), replace=True).mean(axis=1))
    ratio = math.exp(float(values.mean()))
    ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
    dynamic_values = np.array([statistics.mean(v) for v in dynamic_by_prompt.values()])
    dynamic_draws = np.exp(
        rng.choice(dynamic_values, (10000, len(dynamic_values)), replace=True).mean(
            axis=1
        )
    )
    dynamic_ratio = math.exp(float(dynamic_values.mean()))
    dynamic_ci = [
        float(np.quantile(dynamic_draws, 0.025)),
        float(np.quantile(dynamic_draws, 0.975)),
    ]
    within = any(len(v) > 1 for v in trajectories.values())
    return {
        "schema_version": 2,
        "q2_supported": (
            len(winners) > 1
            and within
            and ratio > 1.05
            and ci[0] > 1
            and dynamic_ratio > 1.05
            and dynamic_ci[0] > 1
        ),
        "status": "exploratory development replay; not a deployable speedup claim",
        "global_action": int(global_action),
        "rule": {str(k): int(v) for k, v in rules.items()},
        "features": "historical canonical acceptance >=0.5; context >=256",
        "validation_ratio_after_estimated_overhead": ratio,
        "ci95": ci,
        "validation_ratio_vs_dynamic_after_estimated_overhead": dynamic_ratio,
        "dynamic_ci95": dynamic_ci,
        "validation_prompts": len(by_prompt),
        "validation_checkpoints": len(val),
        "different_hindsight_winners": sorted(winners),
        "predicted_within_request_variation": within,
        "overhead_assumptions": {
            "decision_ms": 0.05,
            "probe_every": 16,
            "transition_amortization_cycles": 4,
        },
        "limitation": "Hindsight action variation is diagnostic, not an achievable bound. Canonical probe history differs from a learned policy history; a passing pilot still needs an online controller evaluation.",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=pathlib.Path, required=True)
    parser.add_argument("--target", type=pathlib.Path, required=True)
    parser.add_argument("--draft", type=pathlib.Path)
    parser.add_argument("--pilot-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    frozen = json.loads((args.pilot_dir / "run_manifest.json").read_text())["frozen"]
    drafter = "ngram" if frozen["draft_sha256"] is None else "model"
    investigate(
        args.exe,
        args.target,
        args.draft,
        args.pilot_dir,
        args.out,
        frozen["config"],
        frozen["gpu_layers"],
        drafter,
    )
