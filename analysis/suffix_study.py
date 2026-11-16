"""Audit the frozen suffix proposal comparison against unchanged baselines."""

import json
from pathlib import Path
import statistics

from analysis.controller import verify
from analysis.report import cells, paired_summary
from bench.common import ROOT, atomic_json, read_jsonl
from scripts.check_metal_profile import MODEL_SHA256, metal_profile

PAIRS = (
    ("suffix1", "target"),
    ("suffix1", "ngram1"),
    ("suffix1", "ngram8"),
    ("ngram1", "target"),
)


def analyze(directory):
    directory = Path(directory)
    verify(directory)
    frozen = json.loads((directory / "run_manifest.json").read_text())["frozen"]
    config = json.loads((ROOT / "configs/suffix-screen-v1.json").read_text())
    corpus = {p["id"]: p for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")}
    if (
        frozen["config"] != config
        or frozen["target_sha256"] != MODEL_SHA256
        or frozen["draft_sha256"] is not None
        or "configuration_binary_sha256" in frozen
        or {p["id"] for p in frozen["prompts"]} != set(config["prompt_ids"])
        or any(p != corpus.get(p["id"]) for p in frozen["prompts"])
    ):
        raise ValueError(
            "Suffix study requires the frozen single-binary grid and pinned model"
        )
    rows = read_jsonl(directory / "requests.jsonl")
    totals = {}
    for row in rows:
        metal_profile(row, False)
        total = totals.setdefault(
            row["configuration_id"],
            {
                key: 0
                for key in (
                    "cycles",
                    "drafted",
                    "accepted",
                    "verified_rows",
                    "generation_ms",
                    "draft_ms",
                    "verify_ms",
                )
            },
        )
        for key in (
            "cycles",
            "drafted",
            "accepted",
            "verified_rows",
            "draft_ms",
            "verify_ms",
        ):
            total[key] += row["execution"][key]
        total["generation_ms"] += row["generation_ms"]
    table = cells(rows)
    warm = {
        key: 1
        / statistics.median(
            row["warm_ms"]
            for row in rows
            if (row["prompt_id"], row["output_cap"], row["configuration_id"]) == key
        )
        for key in table
    }
    result = {
        "schema_version": 1,
        "state": "suffix_screen_complete",
        "records": len(rows),
        "comparisons": {f"{a}/{b}": paired_summary(table, a, b) for a, b in PAIRS},
        "warm_comparisons": {f"{a}/{b}": paired_summary(warm, a, b) for a, b in PAIRS},
        "work_totals": totals,
        "exact_output_ids": True,
        "limitation": "The three development-fit prompts helped select the suffix threshold. This screen is not held-out evidence, even at the longer 128-token cap.",
    }
    atomic_json(directory / "analysis/summary.json", result)
    return result
