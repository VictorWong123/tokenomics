"""Evaluate the frozen suffix5/cap1 proposal on archived development validation.

Offline canonical-output replay only: no inference, latency or held-out claim.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from analysis.controller import verify
from analysis.proposal_replay import replay, suffix_draft, upstream_ngram
from bench.common import ROOT, atomic_json, read_jsonl, sha256, source_fingerprint

IDS = {
    f"development-{category}-{number:02}"
    for category in ("qa", "code", "summary")
    for number in (3, 4)
}
FROZEN_SELECTION_COMMIT = "ce2e032"


def evaluate(directory, label):
    directory = Path(directory)
    verify(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text())["frozen"]
    corpus = {p["id"]: p for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")}
    if {p["id"] for p in manifest["prompts"]} != IDS or any(
        p != corpus.get(p["id"]) for p in manifest["prompts"]
    ):
        raise ValueError("Requires the original six development-validation prompts")
    rows = read_jsonl(directory / "requests.jsonl")
    indexed = {
        (r["prompt_id"], r["output_cap"], r["repetition"], r["configuration_id"]): r
        for r in rows
    }
    outcomes = []
    checks = 0
    for target in rows:
        if target["configuration_id"] != "target" or target["repetition"] != 0:
            continue
        if target["stop"] != "budget" or target["output_cap"] not in (64, 128):
            raise ValueError(
                "Replay currently requires full budget-stopped validation outputs"
            )
        for strategy, cap, propose in (
            ("ngram1", 1, upstream_ngram),
            ("ngram8", 8, upstream_ngram),
            ("suffix5-cap1", 1, lambda h, p, c: suffix_draft(h, p, c, minimum=5)),
        ):
            counts, _ = replay(target["prompt_ids"], target["token_ids"], cap, propose)
            if strategy.startswith("ngram"):
                cid = "fixed1" if cap == 1 else "dynamic"
                for repetition in range(3):
                    baseline = indexed[
                        (target["prompt_id"], target["output_cap"], repetition, cid)
                    ]
                    if baseline["token_ids"] != target["token_ids"] or any(
                        counts[key] != baseline["execution"][key]
                        for key in ("cycles", "drafted", "accepted", "verified_rows")
                    ):
                        raise ValueError(
                            "Canonical replay differs from archived ngram baseline work"
                        )
                    checks += 1
            outcomes.append(
                {
                    "prompt_id": target["prompt_id"],
                    "output_cap": target["output_cap"],
                    "strategy": strategy,
                    **counts,
                }
            )
    if len(outcomes) != 36:
        raise ValueError("Expected twelve prompt/cap cells and three frozen strategies")
    totals = {}
    for row in outcomes:
        total = totals.setdefault(
            row["strategy"],
            {
                key: 0
                for key in (
                    "tokens",
                    "cycles",
                    "drafted",
                    "accepted",
                    "verified_rows",
                    "wasted_rows",
                )
            },
        )
        for key in total:
            total[key] += row[key]
    for total in totals.values():
        saved = total["tokens"] - total["cycles"]
        total["required_fixed_cost_fraction"] = (
            total["wasted_rows"] / (saved + total["wasted_rows"]) if saved else None
        )
    comparisons = []
    for pid in sorted(IDS):
        for cap in (64, 128):
            group = {
                r["strategy"]: r
                for r in outcomes
                if r["prompt_id"] == pid and r["output_cap"] == cap
            }
            suffix = group["suffix5-cap1"]
            comparisons.append(
                {
                    "prompt_id": pid,
                    "output_cap": cap,
                    "suffix_minus_ngram1_cycles": suffix["cycles"]
                    - group["ngram1"]["cycles"],
                    "suffix_minus_ngram1_wasted_rows": suffix["wasted_rows"]
                    - group["ngram1"]["wasted_rows"],
                    "suffix_minus_ngram8_cycles": suffix["cycles"]
                    - group["ngram8"]["cycles"],
                    "suffix_minus_ngram8_wasted_rows": suffix["wasted_rows"]
                    - group["ngram8"]["wasted_rows"],
                }
            )
    return {
        "label": label,
        "baseline_work_checks": checks,
        "manifest_sha256": sha256(directory / "run_manifest.json"),
        "requests_sha256": sha256(directory / "requests.jsonl"),
        "totals": totals,
        "cells": outcomes,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cpu = evaluate(
        ROOT
        / "results/verification/codespace-controller-20261108/attempt-2/validation",
        "linux-cpu-3b",
    )
    archive = ROOT / "results/verification/mac-m3-metal-20261111/validation.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with tarfile.open(archive) as tar:
            seen = set()
            for member in tar.getmembers():
                path = PurePosixPath(member.name)
                if not path.parts or path.parts[0] != "validation" or member.isdir():
                    continue
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not member.isfile()
                    or member.name in seen
                ):
                    raise ValueError("Unsafe archive member")
                seen.add(member.name)
                dest = root / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(member).read())
        mac = evaluate(root / "validation", "m3-metal-05b")
    result = {
        "schema_version": 1,
        "state": "frozen_proposal_validation_replayed",
        "selection_commit": FROZEN_SELECTION_COMMIT,
        "source": source_fingerprint(),
        "mac_archive_sha256": sha256(archive),
        "scope": "Six historical development-validation prompts, caps64/128. Canonical replay assumes exact target outputs for alternate proposal paths. No model inference, measured speedup or reserved held-out evaluation.",
        "strategies": "Frozen suffix minimum5/maximum32/cap1 versus upstream ngram3 caps1/8. No threshold search on validation.",
        "sources": [cpu, mac],
        "speed_claim": None,
    }
    atomic_json(args.out / "validation.json", result)
    for source in result["sources"]:
        print(source["label"], json.dumps(source["totals"], indent=2))


if __name__ == "__main__":
    main()
