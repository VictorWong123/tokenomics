#!/usr/bin/env python3
"""Differential model gate, not a benchmark or proof of unobserved coverage.

Compare the adapter's ordinary target-only decoding with resident target-only,
every fixed cap, both cache paths, dynamic policies and fallback/re-entry.
Keep jobs, outputs, cycle logs and the first divergence, including on failure.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench.common import (
    ROOT,
    atomic_json,
    hardware,
    read_jsonl,
    sha256,
    source_fingerprint,
)
from bench.run import first_mismatch


def variants():
    yield {"configuration_id": "resident", "policy": "target"}
    for cache in ("retain", "upstream"):
        for k in (1, 2, 4, 6, 8):
            yield {
                "configuration_id": f"{cache}-fixed{k}",
                "policy": "fixed",
                "k": k,
                "cache": cache,
            }
        for policy in ("confidence", "acceptance", "script"):
            yield {
                "configuration_id": f"{cache}-{policy}",
                "policy": policy,
                "k": 8,
                "cache": cache,
                "script": [0, 0, 4, 1, 0, 8, 2],
            }


def compare(expected, actual):
    for field in ("prompt_ids", "token_ids"):
        mismatch = first_mismatch(expected[field], actual[field])
        if mismatch is not None:
            return {"field": field, **mismatch}
    if expected["stop"] != actual["stop"]:
        return {"field": "stop", "expected": expected["stop"], "actual": actual["stop"]}
    if actual["generated_tokens"] != len(actual["token_ids"]):
        return {"field": "generated_tokens"}
    return None


def stream_process(command, result_path, log, timeout, on_row):
    """Read only complete flushed records; stop immediately on a failed check."""
    process = subprocess.Popen(command, stdout=log, stderr=log)
    deadline = time.monotonic() + timeout
    stream = None
    pending = b""

    def drain():
        nonlocal stream, pending
        if stream is None and result_path.exists():
            stream = result_path.open("rb")
        if stream is not None:
            pending += stream.read()
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if line.strip():
                    on_row(json.loads(line))
            if len(pending) > 4 * 1024**2:
                raise RuntimeError("oversized incomplete engine record")

    try:
        while True:
            drain()
            code = process.poll()
            if code is not None:
                drain()  # include a last flush racing with the previous read
                if code:
                    raise subprocess.CalledProcessError(code, command)
                if pending.strip():
                    raise RuntimeError("incomplete final engine record")
                return
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(0.05)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if stream is not None:
            stream.close()


def validate_record(a, job, row):
    if any(
        row[key] != job[key] for key in ("prompt_id", "configuration_id", "repetition")
    ):
        raise RuntimeError("output identity/order mismatch")
    expected = {
        "context": a.context,
        "batch": a.batch,
        "threads": a.threads,
        "gpu_layers": a.gpu_layers,
        "sampler": "greedy-temperature-only",
    }
    for key, default in (
        ("policy", "target"),
        ("k", 4),
        ("cache", "retain"),
        ("drafter", "model"),
        ("confidence", 0.5),
        ("max_tokens", 128),
        ("max_cycles", 0),
        ("script", []),
        ("diagnostic_logits", False),
    ):
        expected[key] = job.get(key, default)
    if row["output_cap"] != expected["max_tokens"] or any(
        row["settings"].get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("output execution settings mismatch")


def execute(a, name, jobs, resident, on_row=None):
    job_path = a.out / f"{name}.jobs.jsonl"
    job_path.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    command = [
        str(a.binary.resolve()),
        "--target",
        str(a.target.resolve()),
        "--jobs",
        str(job_path),
        "--out",
        str(a.out / f"{name}.jsonl"),
        "--cycles",
        str(a.out / f"{name}.cycles.jsonl"),
        "--context",
        str(a.context),
        "--batch",
        str(a.batch),
        "--threads",
        str(a.threads),
        "--gpu-layers",
        str(a.gpu_layers),
    ]
    if resident:
        command += ["--draft", str(a.draft.resolve())]
    seen = 0

    def checked_row(row):
        nonlocal seen
        if seen >= len(jobs):
            raise RuntimeError(f"{name}: output identity/order mismatch")
        validate_record(a, jobs[seen], row)
        seen += 1
        on_row(row)

    with (a.out / f"{name}.stderr.log").open("w") as log:
        if on_row is None:
            subprocess.run(
                command, stdout=log, stderr=log, check=True, timeout=a.timeout_seconds
            )
        else:
            stream_process(
                command, a.out / f"{name}.jsonl", log, a.timeout_seconds, checked_row
            )
    rows = read_jsonl(a.out / f"{name}.jsonl")
    if len(rows) != len(jobs):
        raise RuntimeError(f"{name}: incomplete outputs ({len(rows)}/{len(jobs)})")
    for job, row in zip(jobs, rows):
        validate_record(a, job, row)
    return rows


def check_cycles(rows, cycles):
    grouped = collections.defaultdict(list)
    for cycle in cycles:
        grouped[
            (cycle["prompt_id"], cycle["configuration_id"], cycle["repetition"])
        ].append(cycle)
    keys = {
        (row["prompt_id"], row["configuration_id"], row["repetition"]) for row in rows
    }
    if len(keys) != len(rows) or set(grouped) - keys:
        raise RuntimeError("duplicate request identity or orphan cycle records")
    for row in rows:
        key = (row["prompt_id"], row["configuration_id"], row["repetition"])
        records = grouped[key]
        if sum(c["committed"] for c in records) != row["generated_tokens"]:
            raise RuntimeError(f"cycle commit count mismatch: {key}")
        if row["generated_tokens"]:
            expected_cache = len(row["prompt_ids"]) + row["generated_tokens"] - 1
            if (
                row["target_cached"] != expected_cache
                or not 0 <= row["draft_cached"] <= expected_cache
            ):
                raise RuntimeError(f"final cache position mismatch: {key}")


def run(a):
    # A fresh directory preserves every failed/partial attempt. No implicit resume.
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    checked = 0
    first_failure = None
    try:
        if min(a.context, a.batch, a.threads, a.max_tokens, a.timeout_seconds) <= 0:
            raise ValueError("positive settings required")
        prompts = read_jsonl(a.prompts)
        selected = [
            next(
                p
                for p in prompts
                if p["split"] == "development" and p["category"] == cat
            )
            for cat in ("qa", "code", "summary")
        ]
        atomic_json(
            a.out / "manifest.json",
            {
                "schema_version": 1,
                "purpose": "differential correctness; not performance",
                "reference": "Tokenomics ordinary target-only adapter; not exact stock executable",
                "source": source_fingerprint(),
                "hardware": hardware(),
                "binary_sha256": sha256(a.binary),
                "target_sha256": sha256(a.target),
                "draft_sha256": sha256(a.draft),
                "prompts_sha256": sha256(a.prompts),
                "synthetic_fixture": a.synthetic_fixture,
                "settings": {
                    key: getattr(a, key)
                    for key in (
                        "context",
                        "batch",
                        "threads",
                        "gpu_layers",
                        "max_tokens",
                        "timeout_seconds",
                    )
                },
            },
        )
        cases = [
            {"prompt_id": p["id"], "prompt": p["prompt"], "max_tokens": a.max_tokens}
            for p in selected
        ]
        # Empty input is model/tokenizer dependent. Use explicit nonempty text for
        # budget cases; the synthetic suite owns forced EOG/fault injection.
        cases += [
            {"prompt_id": f"budget-{cap}", "prompt": "a", "max_tokens": cap}
            for cap in (0, 1, 2, 7)
        ]
        reference_jobs = [
            {**c, "policy": "target", "configuration_id": "target", "repetition": 0}
            for c in cases
        ]
        references = execute(a, "reference", reference_jobs, False)
        # Use real tokenizer IDs to exercise the exact logical context boundary.
        seed = references[-1]["prompt_ids"]
        boundary = {
            "prompt_id": "logical-context-boundary",
            "prefix_tokens": (seed * (a.context // len(seed) + 1))[: a.context],
            "max_tokens": 7,
        }
        cases.append(boundary)
        references += execute(
            a,
            "boundary-reference",
            [
                {
                    **boundary,
                    "policy": "target",
                    "configuration_id": "target",
                    "repetition": 0,
                }
            ],
            False,
        )
        jobs = [
            {**case, **variant, "repetition": rep}
            for case in cases
            for variant in variants()
            for rep in (0, 1)
        ]
        expected = {r["prompt_id"]: r for r in references}
        with (a.out / "correctness.jsonl").open("w") as stream:

            def check_row(row):
                nonlocal checked, first_failure
                mismatch = compare(expected[row["prompt_id"]], row)
                record = {
                    "prompt_id": row["prompt_id"],
                    "configuration_id": row["configuration_id"],
                    "repetition": row["repetition"],
                    "ok": mismatch is None,
                    "mismatch": mismatch,
                }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                checked += 1
                if mismatch is not None:
                    first_failure = record
                    raise RuntimeError(f"first divergence: {json.dumps(record)}")

            actual = execute(a, "candidate", jobs, True, check_row)
        cycles = read_jsonl(a.out / "candidate.cycles.jsonl")
        check_cycles(actual, cycles)
        coverage = {
            "zero_acceptance": any(
                c["actual_drafts"] > 0 and c["accepted"] == 0 for c in cycles
            ),
            "partial_acceptance": any(
                0 < c["accepted"] < c["actual_drafts"] for c in cycles
            ),
            "full_acceptance": any(
                c["actual_drafts"] > 0 and c["accepted"] == c["actual_drafts"]
                for c in cycles
            ),
            "catchup": any(c["synced"] > 1 for c in cycles),
            "switch": any(c["switched"] for c in cycles),
            "stop_reasons": sorted({r["stop"] for r in actual}),
        }
        result = {
            "schema_version": 1,
            "state": "passed",
            "comparisons": len(actual),
            "synthetic_fixture": a.synthetic_fixture,
            "observed_coverage": coverage,
            "limitation": "Only these inputs/settings tested; absent coverage remains unverified. No speedup evidence.",
        }
        atomic_json(a.out / "status.json", result)
        return result
    except Exception as exc:
        atomic_json(
            a.out / "status.json",
            {
                "schema_version": 1,
                "state": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "checked_comparisons": checked,
                "first_mismatch": first_failure,
            },
        )
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=pathlib.Path, default=ROOT / "build-cpu/tokenomics")
    p.add_argument("--target", type=pathlib.Path, required=True)
    p.add_argument("--draft", type=pathlib.Path, required=True)
    p.add_argument(
        "--prompts", type=pathlib.Path, default=ROOT / "bench/prompts-v1.jsonl"
    )
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--gpu-layers", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    p.add_argument("--synthetic-fixture", action="store_true")
    return p


if __name__ == "__main__":
    try:
        print(json.dumps(run(parser().parse_args()), indent=2))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
