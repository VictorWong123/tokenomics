#!/usr/bin/env python3
"""Curate actual fixture observations, retaining measurements without alteration."""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench.common import read_jsonl, atomic_json, sha256, write_jsonl
from analysis.report import write_report


def curate(source, out):
    manifest = json.loads((source / "run_manifest.json").read_text())
    if not manifest["frozen"].get("identity", {}).get("synthetic_fixture"):
        raise ValueError("only fixture observations are allowed as the default example")
    manifest["curation"] = {
        "source_manifest_sha256": sha256(source / "run_manifest.json"),
        "retained_repetitions": [0],
        "source_repetitions": manifest["frozen"]["config"]["repetitions"],
    }
    manifest["frozen"]["config"]["repetitions"] = 1
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_jsonl(source / "requests.jsonl") if r["repetition"] == 0]
    cycles = []
    seen = set()
    for row in rows:
        original = row["source_cycles"]
        if original not in seen:
            seen.add(original)
            cycles.extend(
                c for c in read_jsonl(source / original) if not c.get("warmup", False)
            )
        row["source_cycles"] = "cycles.jsonl"
    write_jsonl(out / "requests.jsonl", rows)
    write_jsonl(out / "cycles.jsonl", cycles)
    checks = [
        c for c in read_jsonl(source / "correctness.jsonl") if c["repetition"] == 0
    ]
    write_jsonl(out / "correctness.jsonl", checks)
    atomic_json(out / "run_manifest.json", manifest)
    atomic_json(
        out / "complete.json",
        {
            "schema_version": 1,
            "records": len(rows),
            "curated": True,
            "source_requests_sha256": sha256(source / "requests.jsonl"),
            "note": "Observed synthetic-transformer timings, repetition 0 retained; measurements unchanged. Cycle pointers consolidated. Not feasibility evidence.",
        },
    )
    write_report(out, out / "analysis")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=pathlib.Path)
    p.add_argument("out", type=pathlib.Path)
    a = p.parse_args()
    curate(a.source, a.out)
