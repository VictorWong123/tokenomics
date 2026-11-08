"""Resumable balanced blocks. Every block must pass exact token-ID equivalence."""

from __future__ import annotations
import argparse
import json
import pathlib
import random
import subprocess
import time
import re
from bench.common import (
    ROOT,
    atomic_json,
    sha256,
    hardware,
    read_jsonl,
    source_fingerprint,
)


class RunPaused(RuntimeError):
    """A planned pause at a comparison-block boundary, not an invalid run."""


def save_rows(out, requests, correctness):
    for name, rows in (
        ("requests.jsonl", requests),
        ("correctness.jsonl", correctness),
    ):
        tmp = out / (name + ".tmp")
        tmp.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
        tmp.replace(out / name)


def balanced_blocks(prompts, caps, repetitions, configs, seed):
    rng = random.Random(seed)
    blocks = [(p, cap, r) for p in prompts for cap in caps for r in range(repetitions)]
    rng.shuffle(blocks)
    base = list(configs)
    rng.shuffle(base)
    for i, (p, cap, r) in enumerate(blocks):
        shift = i % len(base)
        yield p, cap, r, base[shift:] + base[:shift]


def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"position": i, "reference": x, "actual": y}
    if len(a) != len(b):
        return {
            "position": min(len(a), len(b)),
            "reference_length": len(a),
            "actual_length": len(b),
        }
    return None


def run(
    exe,
    target,
    draft,
    prompts,
    config,
    out,
    gpu_layers=0,
    identity=None,
    *,
    deadline=None,
    max_new_blocks=None,
    executables=None,
):
    """Measure complete blocks with frozen binaries and exact ordinary references.

    An optional per-configuration executable map supports the paired CPU-profile
    component study. It must cover every condition; its ordinary target must
    match exe. Every executable hash participates in resume identity.
    """
    conf_ids = [c["id"] for c in config["configurations"]]
    if executables is not None and set(executables) != set(conf_ids):
        raise ValueError("Each configuration requires an explicit executable")
    if any(
        c.get(key)
        for c in config["configurations"]
        for key in (
            "capture_checkpoints",
            "checkpoint_namespace",
            "resume_checkpoint",
            "prefix_tokens",
            "max_cycles",
            "diagnostic_logits",
        )
    ):
        raise ValueError("diagnostic requests cannot be performance conditions")
    prompt_ids = [p["id"] for p in prompts]
    if (
        not prompts
        or not conf_ids
        or len(set(conf_ids)) != len(conf_ids)
        or len(set(prompt_ids)) != len(prompt_ids)
    ):
        raise ValueError("nonempty unique prompt/configuration IDs required")
    if "target" not in conf_ids or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in conf_ids + prompt_ids
    ):
        raise ValueError("safe IDs and an ordinary target reference are required")
    if (
        config["repetitions"] <= 0
        or not config["output_caps"]
        or any(n <= 0 for n in config["output_caps"])
    ):
        raise ValueError("positive repetitions and output caps required")
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "schema_version": 1,
        "config": config,
        "prompts": prompts,
        "target_sha256": sha256(target),
        "draft_sha256": sha256(draft) if draft else None,
        "binary_sha256": sha256(exe),
        "gpu_layers": gpu_layers,
        "identity": identity or {},
        "code": source_fingerprint(),
    }
    if executables is not None:
        settings["configuration_binary_sha256"] = {
            cid: sha256(path) for cid, path in sorted(executables.items())
        }
        if (
            settings["configuration_binary_sha256"]["target"]
            != settings["binary_sha256"]
        ):
            raise ValueError("The ordinary target executable must match --exe")
    # Reject stale or synthetic calibration before any model execution. Backend
    # identity is checked again on the returned record below.
    from bench.controller import point_from_frozen, validate_binding

    for conf in config["configurations"]:
        if conf.get("policy") == "cost-aware":
            validate_binding(
                conf.get("controller", {}),
                point_from_frozen(
                    settings,
                    conf.get("drafter", "model"),
                    conf.get("cache", "retain"),
                    configuration_id=conf["id"],
                ),
                synthetic_fixture=settings["identity"].get("synthetic_fixture", False),
            )
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["frozen"] != settings:
            raise RuntimeError(
                "Resume refused: binary/models/prompts/configuration changed. Use a new run directory."
            )
        run_id = old["run_id"]
    else:
        run_id = out.name
        atomic_json(
            manifest_path,
            {"run_id": run_id, "frozen": settings, "hardware_start": hardware()},
        )
    requests = []
    correctness = []
    total = len(prompts) * len(config["output_caps"]) * config["repetitions"]
    new_blocks = 0
    longest_block = 0.0
    for index, (prompt, cap, rep, order) in enumerate(
        balanced_blocks(
            prompts,
            config["output_caps"],
            config["repetitions"],
            config["configurations"],
            config["seed"],
        )
    ):
        block_id = f"{prompt['id']}-n{cap}-r{rep}"
        bdir = out / "blocks" / block_id
        done = bdir / "complete.json"
        if done.exists():
            complete = json.loads(done.read_text())
            for path, digest in complete.get("sha256", {}).items():
                if sha256(bdir / path) != digest:
                    raise RuntimeError(
                        "Completed record hash changed; investigate artifact integrity"
                    )
            records = [json.loads((bdir / x).read_text()) for x in complete["records"]]
            requests.extend(records)
            correctness.extend(complete["correctness"])
            continue
        if (max_new_blocks is not None and new_blocks >= max_new_blocks) or (
            deadline is not None and time.monotonic() + 1.25 * longest_block >= deadline
        ):
            save_rows(out, requests, correctness)
            raise RunPaused(
                f"{index}/{total} complete blocks retained; resume unchanged"
            )
        block_start = time.monotonic()
        bdir.mkdir(parents=True, exist_ok=True)
        attempt = len(list(bdir.glob("attempt-*")))
        adir = bdir / f"attempt-{attempt}"
        adir.mkdir()
        records = []
        paths = []
        try:
            for slot, conf in enumerate(order):
                cid = conf["id"]
                meta = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "configuration_id": cid,
                    "prompt_id": prompt["id"],
                    "repetition": rep,
                    "output_cap": cap,
                    "order": slot,
                }
                job = {
                    **conf,
                    **meta,
                    "prompt": prompt["prompt"],
                    "max_tokens": cap,
                    "detailed": conf.get("detailed", config.get("detailed", True)),
                }
                warm = {**job, "warmup": True, "max_tokens": config["warmup_tokens"]}
                warm["prompt"] = config.get("warmup_prompt", job["prompt"])
                if conf["policy"] == "target" and conf.get("resident", False):
                    # Mapping the draft alone does not fault in its weight pages.
                    warm.update(policy="fixed", k=4, drafter="model")
                jobs = adir / f"{cid}.jobs.jsonl"
                jobs.write_text(json.dumps(warm) + "\n" + json.dumps(job) + "\n")
                result_path = adir / f"{cid}.requests.jsonl"
                cycles_path = adir / f"{cid}.cycles.jsonl"
                args = [
                    str(executables[cid] if executables is not None else exe),
                    "--target",
                    str(target),
                    "--jobs",
                    str(jobs),
                    "--out",
                    str(result_path),
                    "--cycles",
                    str(cycles_path),
                    "--context",
                    str(config["context"]),
                    "--batch",
                    str(config["batch"]),
                    "--threads",
                    str(config["threads"]),
                    "--gpu-layers",
                    str(gpu_layers),
                ]
                if (
                    conf["policy"] != "target"
                    and conf.get("drafter", "model") == "model"
                ) or conf.get("resident", False):
                    if not draft:
                        raise RuntimeError("configuration requires draft model")
                    args += ["--draft", str(draft)]
                start = time.monotonic()
                with open(adir / f"{cid}.stderr.txt", "w") as err:
                    proc = subprocess.run(
                        args, stderr=err, stdout=subprocess.PIPE, text=True
                    )
                process_ms = (time.monotonic() - start) * 1000
                if proc.returncode:
                    raise RuntimeError(
                        f"process_failure {cid}: see {adir}/{cid}.stderr.txt"
                    )
                rows = read_jsonl(result_path)
                if len(rows) != 2 or rows[1].get("warmup", False):
                    raise RuntimeError("missing_or_malformed_record")
                row = rows[1]
                if conf.get("policy") == "cost-aware":
                    validate_binding(
                        conf["controller"],
                        point_from_frozen(
                            settings,
                            conf.get("drafter", "model"),
                            conf.get("cache", "retain"),
                            configuration_id=conf["id"],
                        ),
                        synthetic_fixture=settings["identity"].get(
                            "synthetic_fixture", False
                        ),
                        system_info=row["system_info"],
                    )
                row["process_ms"] = process_ms
                row["category"] = prompt["category"]
                row["block_id"] = block_id
                row["attempt"] = attempt
                row["correctness"] = "pending"
                row["source_record"] = str(result_path.relative_to(out))
                row["source_cycles"] = str(cycles_path.relative_to(out))
                p = adir / f"{cid}.json"
                atomic_json(p, row)
                paths.append(str(p.relative_to(bdir)))
                records.append(row)
            references = [x for x in records if x["configuration_id"] == "target"]
            if len(references) != 1:
                raise RuntimeError(
                    "each comparison block requires exactly one ordinary target reference"
                )
            ref = references[0]
            checks = []
            for row in records:
                mismatch = first_mismatch(ref["token_ids"], row["token_ids"])
                prompt_mismatch = first_mismatch(ref["prompt_ids"], row["prompt_ids"])
                ok = (
                    mismatch is None
                    and prompt_mismatch is None
                    and ref["stop"] == row["stop"]
                )
                check = {
                    k: row[k]
                    for k in (
                        "run_id",
                        "configuration_id",
                        "prompt_id",
                        "repetition",
                        "output_cap",
                    )
                }
                check.update(
                    schema_version=1,
                    ok=ok,
                    first_mismatch=mismatch,
                    prompt_mismatch=prompt_mismatch,
                    block_id=block_id,
                    attempt=attempt,
                )
                checks.append(check)
            atomic_json(adir / "correctness.json", checks)
            if not all(x["ok"] for x in checks):
                raise RuntimeError(
                    "token_mismatch: performance gate closed; investigate first mismatch"
                )
            for row, path in zip(records, paths):
                row["correctness"] = "pass"
                atomic_json(bdir / path, row)
            atomic_json(
                done,
                {
                    "records": paths,
                    "correctness": checks,
                    "attempt": attempt,
                    "sha256": {p: sha256(bdir / p) for p in paths},
                },
            )
            requests.extend(records)
            correctness.extend(checks)
        except (Exception, KeyboardInterrupt) as ex:
            atomic_json(
                adir / "failure.json",
                {
                    "schema_version": 1,
                    "block_id": block_id,
                    "attempt": attempt,
                    "reason": str(ex),
                    "hardware": hardware(),
                },
            )
            raise
        print(f"{out.name}: {index+1}/{total} blocks complete ({block_id})", flush=True)
        new_blocks += 1
        longest_block = max(longest_block, time.monotonic() - block_start)
    # Derived concatenations: immutable per-attempt raw records remain the source.
    save_rows(out, requests, correctness)
    # Calibration hashes completed sources. An unchanged resume must preserve
    # their bytes and the original completion-time hardware observation.
    completion = out / "complete.json"
    if completion.exists():
        recorded = json.loads(completion.read_text())
        if recorded.get("blocks") != total or recorded.get("records") != len(requests):
            raise RuntimeError("Completed run receipt differs from frozen grid")
    else:
        atomic_json(
            completion,
            {
                "schema_version": 1,
                "blocks": total,
                "records": len(requests),
                "hardware_end": hardware(),
            },
        )
    return requests


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exe", type=pathlib.Path, required=True)
    p.add_argument("--target", type=pathlib.Path, required=True)
    p.add_argument("--draft", type=pathlib.Path)
    p.add_argument(
        "--config", type=pathlib.Path, default=ROOT / "configs/pilot-v1.json"
    )
    p.add_argument(
        "--prompts", type=pathlib.Path, default=ROOT / "bench/prompts-v1.jsonl"
    )
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--gpu-layers", type=int, default=0)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    prompts = [
        x
        for x in read_jsonl(a.prompts)
        if x["split"] == cfg["split"]
        and (cfg.get("prompt_ids") != "pilot18" or x["pilot"])
    ]
    run(
        a.exe.resolve(),
        a.target.resolve(),
        a.draft.resolve() if a.draft else None,
        prompts,
        cfg,
        a.out,
        a.gpu_layers,
    )
