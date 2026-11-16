"""Audited, paired M3 development screen; no held-out claim."""

import json
from pathlib import Path
import statistics

from analysis.controller import verify
from analysis.report import cells, paired_summary
from bench.common import ROOT, atomic_json, read_jsonl
from scripts.check_metal_profile import MODEL_SHA256, interleaved_profile, metal_profile


SCREENS = {
    "ngram": "metal-short-mv-screen-v1.json",
    "retrieval": "metal-retrieval-screen-v1.json",
}


def screen_configuration(screen, interleaved=False):
    config = json.loads((ROOT / "configs" / SCREENS[screen]).read_text())
    if interleaved:
        config["kernel_variant"] = "interleaved"
        config["phase"] = config["phase"].replace("metal_", "metal_interleaved_", 1)
        for condition in config["configurations"]:
            if condition["kernel_profile"] == "short-mv":
                condition["kernel_profile"] = "interleaved"
                condition["id"] = condition["id"].replace("short-mv", "interleaved")
    return config


def analyze(directory, *, screen="ngram", interleaved=False):
    directory = Path(directory)
    verify(directory)
    frozen = json.loads((directory / "run_manifest.json").read_text())["frozen"]
    expected = screen_configuration(screen, interleaved)
    candidate = "interleaved" if interleaved else "short-mv"
    profile_check = interleaved_profile if interleaved else metal_profile
    corpus = {p["id"]: p for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")}
    if (
        frozen["config"] != expected
        or frozen["target_sha256"] != MODEL_SHA256
        or frozen["draft_sha256"] is not None
        or {p["id"] for p in frozen["prompts"]} != set(expected["prompt_ids"])
        or any(p != corpus.get(p["id"]) for p in frozen["prompts"])
    ):
        raise ValueError(
            "Metal screen requires the frozen development grid and pinned model"
        )
    confs = {c["id"]: c for c in expected["configurations"]}
    binaries = frozen.get("configuration_binary_sha256", {})
    if set(binaries) != set(confs):
        raise ValueError("Missing paired executable identity")
    for profile in ("guarded", candidate):
        if (
            len(
                {
                    binaries[cid]
                    for cid, c in confs.items()
                    if c["kernel_profile"] == profile
                }
            )
            != 1
        ):
            raise ValueError("Profile changed executable across conditions")
    if binaries["target"] == binaries[f"{candidate}-target"]:
        raise ValueError("Baseline and candidate must be distinct executables")
    rows = read_jsonl(directory / "requests.jsonl")
    groups = {}
    work_keys = ("cycles", "drafted", "accepted", "verified_rows", "synced_tokens")
    totals = {}
    for row in rows:
        cid = row["configuration_id"]
        profile_check(row, confs[cid]["kernel_profile"] == candidate)
        key = row["prompt_id"], row["output_cap"], row["repetition"]
        groups.setdefault(key, {})[cid] = row
        total = totals.setdefault(
            cid,
            {
                key: 0
                for key in (
                    *work_keys,
                    "generation_ms",
                    "verify_ms",
                    "prompt_target_ms",
                )
            },
        )
        for key in (*work_keys, "verify_ms"):
            total[key] += row["execution"][key]
        for key in ("generation_ms", "prompt_target_ms"):
            total[key] += row[key]
    pairs = [(f"{candidate}-target", "target")]
    same_policy = [(f"{candidate}-target", "target")]
    for k in (1, 4, 8):
        same_policy.append((f"{candidate}-fixed{k}", f"guarded-fixed{k}"))
        pairs += [
            (f"{candidate}-fixed{k}", b)
            for b in (f"guarded-fixed{k}", "target", f"{candidate}-target")
        ]
        pairs.append((f"guarded-fixed{k}", "target"))
    if screen == "retrieval":
        for k in (1, 4, 8):
            suffix_candidate = f"{candidate}-suffix{k}"
            same_policy.append((suffix_candidate, f"guarded-suffix{k}"))
            pairs += [
                (suffix_candidate, baseline)
                for baseline in (
                    f"guarded-suffix{k}",
                    "target",
                    f"{candidate}-target",
                    "guarded-fixed8",
                    f"{candidate}-fixed8",
                    f"{candidate}-suffix1",
                )
                if baseline != suffix_candidate
            ]
    for group in groups.values():
        for a, b in same_policy:
            if any(
                group[a]["execution"][key] != group[b]["execution"][key]
                for key in work_keys
            ):
                raise ValueError("Paired policies did different inference work")
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
    summary = {
        "schema_version": 1,
        "state": (
            f"metal_interleaved_{screen}_comparison_complete"
            if interleaved
            else (
                "metal_short_mv_comparison_complete"
                if screen == "ngram"
                else "metal_retrieval_comparison_complete"
            )
        ),
        "screen": screen,
        "kernel_variant": candidate,
        "records": len(rows),
        "comparisons": {f"{a}/{b}": paired_summary(table, a, b) for a, b in pairs},
        "warm_comparisons": {f"{a}/{b}": paired_summary(warm, a, b) for a, b in pairs},
        "work_totals": totals,
        "cross_profile_exact": True,
        "same_work_within_each_policy": True,
        "limitation": "Three development prompts, cap128, three repeats. These fit prompts informed suffix threshold/cap exploration. This is a component screen, not held-out or controller evidence.",
    }
    atomic_json(directory / "analysis/summary.json", summary)
    return summary
