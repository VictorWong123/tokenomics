#!/usr/bin/env python3
"""Cross-binary Metal gate. Raw failures are retained; this is not a benchmark."""

import argparse
from functools import partial
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import ROOT, atomic_json, read_jsonl, sha256, source_fingerprint
from bench.model_identity import (
    DEFAULT_MODEL,
    MODEL_PINS,
    model_identity,
    require_model_variant,
)
from scripts.check_models import check_cycles, compare, execute, validate_record

REFERENCE_MARKER = "TOKENOMICS_METAL_VERIFY_SINGLE_ROW=ON"
CANDIDATE_MARKER = "TOKENOMICS_METAL_SHORT_MV=ON"
MODEL_SHA256 = model_identity()["sha256"]
INTERLEAVE_MARKER = "TOKENOMICS_METAL_INTERLEAVE=ON"
Q8_GROUP_MARKER = "TOKENOMICS_METAL_Q8_GROUP=ON"
Q8_SHARED_MARKER = "TOKENOMICS_METAL_Q8_SHARED=ON"
DIRECT_MARKER = "TOKENOMICS_GREEDY_DIRECT=ON"
GREEDY_MARKER = "TOKENOMICS_GREEDY_NO_TEMP=ON"


def metal_profile(
    row,
    candidate,
    *,
    interleave=False,
    group_q8=False,
    shared_q8=False,
    greedy=False,
    direct=False,
    model_id=DEFAULT_MODEL,
):
    identity = require_model_variant(model_id, grouped_q8=group_q8 or shared_q8)
    offloaded = identity["layers"] + 1
    info = row["system_info"]
    marker, excluded = (
        (CANDIDATE_MARKER, REFERENCE_MARKER)
        if candidate
        else (REFERENCE_MARKER, CANDIDATE_MARKER)
    )
    if (
        (Q8_GROUP_MARKER in info) != (candidate and group_q8)
        or (Q8_SHARED_MARKER in info) != (candidate and shared_q8)
        or (GREEDY_MARKER in info) != greedy
        or (DIRECT_MARKER in info) != direct
        or (INTERLEAVE_MARKER in info) != (candidate and interleave)
        or "TOKENOMICS_LOGIT_COLLECTOR=ON" in info
        or marker not in info
        or excluded in info
        or "devices=MTL0" not in info
        or f"offloaded {offloaded}/{offloaded} layers to GPU" not in info
        or row["settings"]["gpu_layers"] != 99
    ):
        raise ValueError(
            f"Expected fully offloaded {model_id} Metal profile was not executed"
        )


def interleaved_profile(row, candidate, *, model_id=DEFAULT_MODEL):
    metal_profile(row, candidate, interleave=True, model_id=model_id)


def q8_group_profile(row, candidate, *, model_id=DEFAULT_MODEL):
    metal_profile(row, candidate, group_q8=True, model_id=model_id)


def q8_shared_profile(row, candidate, *, model_id=DEFAULT_MODEL):
    metal_profile(row, candidate, group_q8=True, shared_q8=True, model_id=model_id)


def greedy_profile(row, candidate, *, model_id=DEFAULT_MODEL):
    metal_profile(row, False, greedy=candidate, model_id=model_id)
    if (GREEDY_MARKER in row["system_info"]) != candidate:
        raise ValueError("Greedy sampler marker differs from expected profile")


def check_direct_execution(row):
    info = row["system_info"]
    eligibility = re.search(r"(?:^|; )raw_greedy_eligible=(\d+)(?:;|$)", info)
    calls = re.search(r"(?:^|; )raw_greedy_calls=(\d+)(?:;|$)", info)
    if (
        DIRECT_MARKER not in info
        or not eligibility
        or not calls
        or eligibility[1] != "1"
    ):
        raise ValueError("Direct greedy path was not eligible/executed")
    count = int(calls[1])
    work = row["execution"]
    if not work["cycles"] <= count <= work["verified_rows"] or (
        not work["cycles"] and count
    ):
        raise ValueError("Direct greedy call count differs from verified work")


def direct_profile(row, candidate, *, model_id=DEFAULT_MODEL):
    metal_profile(row, False, greedy=candidate, direct=candidate, model_id=model_id)
    if candidate:
        check_direct_execution(row)


def variants(drafter="ngram"):
    yield {"configuration_id": "target", "policy": "target"}
    for cache in ("retain", "upstream"):
        for k in (1, 2, 4, 6, 8):
            yield {
                "configuration_id": f"{cache}-fixed{k}",
                "policy": "fixed",
                "drafter": drafter,
                "cache": cache,
                "k": k,
            }
        yield {
            "configuration_id": f"{cache}-script",
            "policy": "script",
            "drafter": drafter,
            "cache": cache,
            "k": 8,
            "script": [0, 4, 1, 8, 2, 0, 6],
        }
    for k in (4, 8):
        yield {
            "configuration_id": f"minimal-fixed{k}",
            "policy": "fixed",
            "drafter": drafter,
            "k": k,
            "detailed": False,
        }


GATE_STAGES = (
    "smoke-reference",
    "smoke-guarded",
    "smoke-candidate",
    "reference",
    "boundary-reference",
    "candidate",
)
KV_BOUNDARY_LENGTHS = (248, 255, 256, 504, 511, 512)
PREFILL_SHAPE_LENGTHS = (3, 4, 9, 10, 515, 521, 522)


def verify_gate(directory):
    """Check a completed gate's saved payload without running inference."""
    directory = Path(directory)
    receipt = json.loads((directory / "complete.json").read_text())
    expected = {"manifest.json", "correctness.jsonl", "status.json"}
    expected |= {
        f"{stage}.{suffix}"
        for stage in GATE_STAGES
        for suffix in ("jobs.jsonl", "jsonl", "cycles.jsonl", "stderr.log")
    }
    if receipt.get("schema_version") != 1 or set(receipt.get("sha256", {})) != expected:
        raise ValueError("Missing or unexpected gate receipt payload")
    for name, digest in receipt["sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Gate payload changed: {name}")
    status = json.loads((directory / "status.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    gate_version = manifest.get("gate_version", 1)
    if gate_version not in (1, 2):
        raise ValueError("Unsupported gate coverage version")
    lengths = KV_BOUNDARY_LENGTHS + (PREFILL_SHAPE_LENGTHS if gate_version == 2 else ())
    expected_count = (
        11 + sum(length < manifest["settings"]["context"] for length in lengths)
    ) * 30
    if (
        status.get("state") != "passed"
        or status.get("candidate_cases") != expected_count
    ):
        raise ValueError(
            "Expected a complete passing gate including KV padding boundaries"
        )
    settings = SimpleNamespace(**manifest["settings"])
    rows = {}
    for stage in GATE_STAGES:
        jobs = read_jsonl(directory / f"{stage}.jobs.jsonl")
        rows[stage] = read_jsonl(directory / f"{stage}.jsonl")
        if len(jobs) != len(rows[stage]):
            raise ValueError("Gate request/output count mismatch")
        for job, row in zip(jobs, rows[stage]):
            validate_record(settings, job, row)
    expected_rows = {
        row["prompt_id"]: row
        for stage in ("reference", "boundary-reference")
        for row in rows[stage]
    }
    for stage in ("smoke-reference", "smoke-guarded", "smoke-candidate", "candidate"):
        for row in rows[stage]:
            if compare(expected_rows[row["prompt_id"]], row) is not None:
                raise ValueError("Saved gate output differs from ordinary reference")
    if len(rows["candidate"]) != expected_count:
        raise ValueError("Missing candidate gate requests")
    check_cycles(
        [
            row
            for row in rows["candidate"]
            if not row["configuration_id"].startswith("minimal-")
        ],
        read_jsonl(directory / "candidate.cycles.jsonl"),
    )
    audit = read_jsonl(directory / "correctness.jsonl")
    if len(audit) != status["checked_comparisons"] or any(
        row["mismatch"] is not None for row in audit
    ):
        raise ValueError("Gate comparison audit is incomplete or failing")
    return {
        "state": "verified",
        "candidate_cases": expected_count,
        "gate_version": gate_version,
        "receipt_sha256": sha256(directory / "complete.json"),
        "manifest": manifest,
    }


def gate(
    reference,
    candidate,
    model,
    out,
    *,
    context=2048,
    threads=2,
    gpu_layers=99,
    profile_validator=metal_profile,
    drafter="ngram",
    model_id=DEFAULT_MODEL,
):
    model_identity(model_id)
    if drafter not in ("ngram", "suffix"):
        raise ValueError("This model-free gate supports ngram or suffix drafting")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    a = SimpleNamespace(
        binary=reference,
        target=model,
        out=out,
        context=context,
        batch=512,
        threads=threads,
        gpu_layers=gpu_layers,
        timeout_seconds=3600,
    )
    checked = 0
    failures = []
    source = source_fingerprint()
    atomic_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "purpose": "cross-profile correctness, not performance",
            "gate_version": 2,
            "source": source,
            "binary_sha256": {
                "reference": sha256(reference),
                "candidate": sha256(candidate),
            },
            "model_sha256": sha256(model),
            "model_id": model_id,
            "drafter": drafter,
            "settings": {
                "context": context,
                "batch": a.batch,
                "threads": threads,
                "gpu_layers": gpu_layers,
            },
        },
    )
    try:
        corpus = {p["id"]: p for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")}
        ids = [
            "development-code-03",
            "development-qa-01",
            "development-code-01",
            "development-summary-01",
        ]
        cases = [
            {"prompt_id": pid, "prompt": corpus[pid]["prompt"], "max_tokens": 128}
            for pid in ids
        ]
        cases += [
            {"prompt_id": f"budget-{cap}", "prompt": "a", "max_tokens": cap}
            for cap in (0, 1, 2, 7)
        ]
        expected = {}
        with (out / "correctness.jsonl").open("w") as audit:

            def check(row, is_candidate):
                nonlocal checked
                profile_validator(row, is_candidate)
                mismatch = compare(expected[row["prompt_id"]], row)
                record = {
                    "prompt_id": row["prompt_id"],
                    "configuration_id": row["configuration_id"],
                    "repetition": row["repetition"],
                    "candidate": is_candidate,
                    "mismatch": mismatch,
                }
                audit.write(json.dumps(record) + "\n")
                audit.flush()
                checked += 1
                if mismatch is not None:
                    failures.append(record)
                    raise RuntimeError(
                        "Cross-profile divergence; see correctness.jsonl and raw cycle logits"
                    )

            # Fail quickly on the original M3 regression, before the wider gate.
            smoke = {
                **cases[0],
                "configuration_id": "target",
                "policy": "target",
                "repetition": 0,
                "diagnostic_logits": True,
            }
            rows = execute(a, "smoke-reference", [smoke], False)
            profile_validator(rows[0], False)
            if sha256(model) == MODEL_SHA256 and rows[0]["generated_tokens"] != 128:
                raise RuntimeError(
                    "Original M3 regression did not reach its full 128-token cap"
                )
            expected[cases[0]["prompt_id"]] = rows[0]
            smoke_jobs = [
                {
                    **smoke,
                    "configuration_id": f"fixed{k}",
                    "policy": "fixed",
                    "drafter": drafter,
                    "k": k,
                }
                for k in (4, 8)
            ]
            execute(
                a, "smoke-guarded", smoke_jobs, False, lambda row: check(row, False)
            )
            a.binary = candidate
            execute(
                a,
                "smoke-candidate",
                [smoke, *smoke_jobs],
                False,
                lambda row: check(row, True),
            )

            a.binary = reference
            jobs = [
                {
                    **case,
                    "configuration_id": "target",
                    "policy": "target",
                    "repetition": 0,
                }
                for case in cases
            ]
            references = execute(a, "reference", jobs, False)
            for row in references:
                profile_validator(row, False)
                if row["prompt_id"] in expected:
                    check(row, False)
                expected[row["prompt_id"]] = row
            seed = references[-1]["prompt_ids"]
            # Full-context and short-tail cases use actual tokenizer IDs.
            boundary = [
                {
                    "prompt_id": f"context-tail-{room}",
                    "prefix_tokens": (seed * (context // len(seed) + 1))[
                        : context - room
                    ],
                    "max_tokens": 16,
                }
                for room in (0, 3, 9)
            ]
            boundary += [
                {
                    "prompt_id": f"kv-padding-{length}",
                    "prefix_tokens": (seed * (length // len(seed) + 1))[:length],
                    "max_tokens": 16,
                }
                for length in KV_BOUNDARY_LENGTHS
                if length < context
            ]
            boundary += [
                {
                    "prompt_id": f"prefill-shape-{length}",
                    "prefix_tokens": (seed * (length // len(seed) + 1))[:length],
                    "max_tokens": 16,
                }
                for length in PREFILL_SHAPE_LENGTHS
                if length < context
            ]
            rows = execute(
                a,
                "boundary-reference",
                [
                    {
                        **case,
                        "configuration_id": "target",
                        "policy": "target",
                        "repetition": 0,
                    }
                    for case in boundary
                ],
                False,
            )
            for row in rows:
                profile_validator(row, False)
                expected[row["prompt_id"]] = row
            cases += boundary
            jobs = [
                {**case, **variant, "repetition": repetition}
                for case in cases
                for variant in variants(drafter)
                for repetition in (0, 1)
            ]
            a.binary = candidate
            actual = execute(a, "candidate", jobs, False, lambda row: check(row, True))
        # Detailed and minimal paths intentionally differ in cycle logging.
        cycles = read_jsonl(out / "candidate.cycles.jsonl")
        check_cycles(
            [r for r in actual if not r["configuration_id"].startswith("minimal-")],
            cycles,
        )
        coverage = {
            "drafted": sum(r["execution"]["drafted"] for r in actual),
            "accepted": sum(r["execution"]["accepted"] for r in actual),
            "stop_reasons": sorted({r["stop"] for r in actual}),
        }
        if not coverage["drafted"] or not coverage["accepted"]:
            raise RuntimeError("Gate did not exercise nonzero proposals and acceptance")
        if source_fingerprint() != source:
            raise RuntimeError("Source changed during the gate")
        result = {
            "schema_version": 1,
            "state": "passed",
            "checked_comparisons": checked,
            "candidate_cases": len(actual),
            "coverage": coverage,
            "limitation": "Only this model, inputs, settings and observed paths are covered. No speed claim.",
        }
        atomic_json(out / "status.json", result)
        payload = {"manifest.json", "correctness.jsonl", "status.json"}
        payload |= {
            f"{stage}.{suffix}"
            for stage in GATE_STAGES
            for suffix in ("jobs.jsonl", "jsonl", "cycles.jsonl", "stderr.log")
        }
        atomic_json(
            out / "complete.json",
            {
                "schema_version": 1,
                "sha256": {name: sha256(out / name) for name in sorted(payload)},
            },
        )
        verify_gate(out)
        return result
    except Exception as exc:
        atomic_json(
            out / "status.json",
            {
                "schema_version": 1,
                "state": "failed",
                "checked_comparisons": checked,
                "first_mismatch": failures[0] if failures else None,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", choices=tuple(MODEL_PINS), default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--candidate-profile",
        choices=(
            "short-mv",
            "interleaved",
            "q8-group",
            "q8-shared",
            "greedy",
            "direct",
        ),
        default="short-mv",
    )
    args = parser.parse_args()
    identity = require_model_variant(
        args.model_id, grouped_q8=args.candidate_profile in ("q8-group", "q8-shared")
    )
    if sha256(args.model) != identity["sha256"]:
        parser.error(f"Requires the pinned {args.model_id} Q4_K_M model")
    gate(
        args.reference.resolve(),
        args.candidate.resolve(),
        args.model.resolve(),
        args.out,
        model_id=args.model_id,
        profile_validator=partial(
            {
                "short-mv": metal_profile,
                "interleaved": interleaved_profile,
                "q8-group": q8_group_profile,
                "q8-shared": q8_shared_profile,
                "greedy": greedy_profile,
                "direct": direct_profile,
            }[args.candidate_profile],
            model_id=args.model_id,
        ),
    )


if __name__ == "__main__":
    main()
