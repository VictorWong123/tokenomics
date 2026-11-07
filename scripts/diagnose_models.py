#!/usr/bin/env python3
"""Targeted exact-token divergence diagnosis. Never a performance benchmark."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import (
    ROOT,
    atomic_json,
    hardware,
    read_jsonl,
    sha256,
    source_fingerprint,
)
from scripts.check_models import check_cycles, compare, execute


def build_configuration(binary):
    """Record selected build switches without copying arbitrary cache values."""
    keys = {
        "CMAKE_BUILD_TYPE",
        "TOKENOMICS_CPU_BATCH_INVARIANT",
        "GGML_NATIVE",
        "GGML_CPU_REPACK",
        "GGML_LLAMAFILE",
        "GGML_BLAS",
        "GGML_METAL",
        "GGML_CUDA",
        "GGML_AVX",
        "GGML_AVX2",
        "GGML_FMA",
        "GGML_F16C",
    }
    cache = Path(binary).resolve().parent / "CMakeCache.txt"
    values = {key: None for key in sorted(keys)}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            key = line.split(":", 1)[0]
            if key in keys and "=" in line:
                values[key] = line.split("=", 1)[1]
    return {"cache_available": cache.is_file(), "values": values}


def probe_at(
    cycles, configuration, repetition, output_index, prompt_id="development-qa-00"
):
    matches = []
    for cycle in cycles:
        if cycle["prompt_id"] != prompt_id or (
            cycle["configuration_id"],
            cycle["repetition"],
        ) != (
            configuration,
            repetition,
        ):
            continue
        for probe in cycle.get("diagnostic_logits", []):
            if (
                probe["row"] < cycle["committed"]
                and probe["output_index"] == output_index
            ):
                matches.append(
                    {
                        **probe,
                        "margin": probe["top_logit"] - probe["second_logit"],
                        "cycle_begin": cycle["diagnostic_logits"][0]["output_index"],
                        "actual_drafts": cycle["actual_drafts"],
                        "context_tokens": cycle["context_tokens"],
                        "target_cached": cycle["target_cached"],
                    }
                )
    if len(matches) != 1:
        raise RuntimeError("missing or ambiguous committed diagnostic row")
    return matches[0]


def diagnose(target, draft, out, binary=None, threads=2, synthetic_fixture=False):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    a = SimpleNamespace(
        binary=Path(binary or ROOT / "build-cpu/tokenomics"),
        target=Path(target),
        draft=Path(draft),
        out=out,
        context=2048,
        batch=512,
        threads=threads,
        gpu_layers=0,
        timeout_seconds=600,
    )
    try:
        prompt = next(
            p
            for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")
            if p["id"] == "development-qa-00" and p["split"] == "development"
        )
        atomic_json(
            out / "manifest.json",
            {
                "schema_version": 1,
                "purpose": "diagnosis only; no timing interpretation",
                "synthetic_fixture": synthetic_fixture,
                "source": source_fingerprint(),
                "hardware": hardware(),
                "target_sha256": sha256(a.target),
                "draft_sha256": sha256(a.draft),
                "binary_sha256": sha256(a.binary),
                "build_configuration": build_configuration(a.binary),
                "prompt_id": prompt["id"],
                "settings": {
                    "context": a.context,
                    "batch": a.batch,
                    "threads": threads,
                    "max_tokens": 32,
                    "diagnostic_logits": True,
                },
            },
        )
        base = {
            "prompt_id": prompt["id"],
            "prompt": prompt["prompt"],
            "max_tokens": 32,
            "diagnostic_logits": True,
        }
        reference = execute(
            a,
            "reference",
            [
                {
                    **base,
                    "configuration_id": "target",
                    "policy": "target",
                    "repetition": rep,
                }
                for rep in (0, 1)
            ],
            False,
        )
        variants = [{"configuration_id": "resident", "policy": "target"}]
        variants += [
            {
                "configuration_id": f"{cache}-fixed{k}",
                "policy": "fixed",
                "k": k,
                "cache": cache,
            }
            for cache in ("retain", "upstream")
            for k in (1, 4, 6)
        ]
        jobs = [
            {**base, **variant, "repetition": rep}
            for variant in variants
            for rep in (0, 1)
        ]
        actual = execute(a, "full", jobs, True)
        ref_cycles = read_jsonl(out / "reference.cycles.jsonl")
        full_cycles = read_jsonl(out / "full.cycles.jsonl")
        check_cycles(reference, ref_cycles)
        check_cycles(actual, full_cycles)
        comparisons = []
        checkpoints = [p for p in (8, 29) if p < len(reference[0]["token_ids"])]
        for row in actual:
            expected = reference[row["repetition"]]
            mismatch = compare(expected, row)
            record = {
                "configuration_id": row["configuration_id"],
                "repetition": row["repetition"],
                "prompt_id": row["prompt_id"],
                "mismatch": mismatch,
            }
            if mismatch and mismatch["field"] == "token_ids" and "position" in mismatch:
                position = mismatch["position"]
                if position < min(len(expected["token_ids"]), len(row["token_ids"])):
                    ref = probe_at(ref_cycles, "target", row["repetition"], position)
                    cur = probe_at(
                        full_cycles,
                        row["configuration_id"],
                        row["repetition"],
                        position,
                    )
                    record.update(
                        reference_probe=ref,
                        candidate_probe=cur,
                        probe_matches_emitted=(
                            ref["top_id"] == expected["token_ids"][position]
                            and cur["top_id"] == row["token_ids"][position]
                        ),
                    )
                    if cur["cycle_begin"] not in checkpoints and len(checkpoints) < 8:
                        checkpoints.append(cur["cycle_begin"])
            comparisons.append(record)
        # Both resident contexts are freshly prefetched from the same canonical
        # token prefix for every action, including action 0. This removes prior
        # incremental KV history; it does not reproduce that history's rounding.
        replay_jobs = []
        for checkpoint in checkpoints:
            prefix = reference[0]["prompt_ids"] + reference[0]["token_ids"][:checkpoint]
            for k in (0, 4, 6):
                replay_jobs.append(
                    {
                        "prompt_id": prompt["id"],
                        "prefix_tokens": prefix,
                        "configuration_id": f"prefix{checkpoint}-action{k}",
                        "policy": "target" if k == 0 else "fixed",
                        "k": k or 4,
                        "max_tokens": 8,
                        "max_cycles": 1,
                        "repetition": 0,
                        "checkpoint": checkpoint,
                        "diagnostic_logits": True,
                    }
                )
        replay = execute(a, "replay", replay_jobs, True) if replay_jobs else []
        replay_cycles = read_jsonl(out / "replay.cycles.jsonl") if replay_jobs else []
        check_cycles(replay, replay_cycles)
        probes = []
        for row in replay:
            checkpoint = row["checkpoint"]
            probe = probe_at(replay_cycles, row["configuration_id"], 0, 0)
            probes.append(
                {
                    "configuration_id": row["configuration_id"],
                    "checkpoint": checkpoint,
                    "first_token": row["token_ids"][0],
                    "canonical_token": reference[0]["token_ids"][checkpoint],
                    "probe_matches_emitted": probe["top_id"] == row["token_ids"][0],
                    "probe": probe,
                }
            )
        result = {
            "schema_version": 1,
            "state": "diagnostic_complete",
            "synthetic_fixture": synthetic_fixture,
            "correctness_supported": False,
            "reference_reset_equal": compare(reference[0], reference[1]) is None,
            "full_comparisons": comparisons,
            "replay_probes": probes,
            "limitation": "Controlled fresh-prefix probes distinguish batch shape from history, but do not alone prove the cause of retained-cache differences. No performance claim.",
        }
        atomic_json(out / "status.json", result)
        return result
    except Exception as exc:
        atomic_json(
            out / "status.json",
            {
                "schema_version": 1,
                "state": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--binary", type=Path)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--synthetic-fixture", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            diagnose(
                a.target, a.draft, a.out, a.binary, a.threads, a.synthetic_fixture
            ),
            indent=2,
        )
    )
