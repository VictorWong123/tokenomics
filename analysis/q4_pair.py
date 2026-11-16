"""Audit the frozen Q4 pair component study against unchanged ordinary decode."""

import json
from pathlib import Path
import statistics

from analysis.controller import verify
from analysis.report import cells, paired_summary
from bench.common import ROOT, atomic_json, read_jsonl

MARKER = "TOKENOMICS_CPU_Q4_PAIR=ON"
PAIRS = [("pair-target", "target")]
for cap in (1, 4, 8):
    PAIRS.extend(
        [
            (f"pair-fixed{cap}", f"guarded-fixed{cap}"),
            (f"pair-fixed{cap}", "target"),
            (f"pair-fixed{cap}", "pair-target"),
            (f"guarded-fixed{cap}", "target"),
        ]
    )


def analyze(directory, out=None):
    directory = Path(directory)
    verify(directory)
    frozen = json.loads((directory / "run_manifest.json").read_text())["frozen"]
    cfg = frozen["config"]
    expected = json.loads((ROOT / "configs/q4-pair-screen-v1.json").read_text())
    if cfg != expected or {p["id"] for p in frozen["prompts"]} != set(
        expected["prompt_ids"]
    ):
        raise ValueError("Q4 pair study requires its complete frozen development grid")
    if not frozen.get("identity", {}).get("synthetic_fixture", False):
        corpus = {
            p["id"]: p
            for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")
            if p["id"] in expected["prompt_ids"]
        }
        if any(p != corpus.get(p["id"]) for p in frozen["prompts"]):
            raise ValueError("Trained Q4 pair prompts differ from the frozen corpus")
    rows = read_jsonl(directory / "requests.jsonl")
    confs = {c["id"]: c for c in cfg["configurations"]}
    binary_map = frozen.get("configuration_binary_sha256", {})
    if set(binary_map) != set(confs):
        raise ValueError("Missing paired executable identity")
    for profile in ("guarded", "q4-pair"):
        if (
            len({binary_map[c] for c in confs if confs[c]["kernel_profile"] == profile})
            != 1
        ):
            raise ValueError("Profile changed executable across conditions")
    groups = {}
    totals = {}
    work_keys = ("cycles", "drafted", "accepted", "verified_rows", "synced_tokens")
    for row in rows:
        cid = row["configuration_id"]
        if (
            (MARKER in row["system_info"])
            != (confs[cid]["kernel_profile"] == "q4-pair")
            or "TOKENOMICS_CPU_BATCH_INVARIANT=ON" not in row["system_info"]
            or "TOKENOMICS_CPU_PACKED_VERIFY=ON" in row["system_info"]
            or row["settings"]["gpu_layers"] != 0
        ):
            raise ValueError("Q4 pair execution profile differs from frozen condition")
        key = row["prompt_id"], row["output_cap"], row["repetition"]
        groups.setdefault(key, {})[cid] = row
        costs = totals.setdefault(
            cid, {k: 0 for k in work_keys + ("generation_ms", "verify_ms")}
        )
        for k in work_keys + ("verify_ms",):
            costs[k] += row["execution"][k]
        costs["generation_ms"] += row["generation_ms"]
    for group in groups.values():
        for candidate, baseline in [("pair-target", "target")] + [
            (f"pair-fixed{k}", f"guarded-fixed{k}") for k in (1, 4, 8)
        ]:
            if any(
                group[candidate]["execution"][k] != group[baseline]["execution"][k]
                for k in work_keys
            ):
                raise ValueError(
                    "Paired kernel conditions did different inference work"
                )
    table = cells(rows)
    warm = {
        key: 1
        / statistics.median(
            r["warm_ms"]
            for r in rows
            if (r["prompt_id"], r["output_cap"], r["configuration_id"]) == key
        )
        for key in table
    }
    summary = {
        "schema_version": 1,
        "state": "q4_pair_comparison_complete",
        "records": len(rows),
        "synthetic_fixture": frozen.get("identity", {}).get("synthetic_fixture", False),
        "comparisons": {f"{a}/{b}": paired_summary(table, a, b) for a, b in PAIRS},
        "warm_comparisons": {f"{a}/{b}": paired_summary(warm, a, b) for a, b in PAIRS},
        "work_totals": totals,
        "cross_profile_exact": True,
        "same_work_within_each_policy": True,
        "limitation": "Three development prompts, one cap, three repeats; component screen, not held-out/controller evidence.",
    }
    atomic_json(Path(out) if out else directory / "analysis/summary.json", summary)
    return summary
