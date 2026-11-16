"""Audited, paired M3 development screen; no held-out claim."""

from functools import partial
import json
from pathlib import Path
import statistics

from analysis.controller import verify
from analysis.report import cells, paired_summary
from bench.common import ROOT, atomic_json, read_jsonl
from bench.model_identity import DEFAULT_MODEL, model_identity, require_model_variant
from scripts.check_metal_profile import (
    MODEL_SHA256,
    direct_profile,
    interleaved_profile,
    metal_profile,
    q8_group_profile,
    q8_shared_profile,
)


SCREENS = {
    "ngram": "metal-short-mv-screen-v1.json",
    "retrieval": "metal-retrieval-screen-v1.json",
}


def kernel_variant(interleaved=False, group_q8=False, shared_q8=False, direct=False):
    if sum(bool(value) for value in (interleaved, group_q8, shared_q8, direct)) > 1:
        raise ValueError("Select only one isolated Metal/sampler experiment")
    if direct:
        return "direct"
    return (
        "q8-shared"
        if shared_q8
        else "q8-group" if group_q8 else "interleaved" if interleaved else "short-mv"
    )


def screen_configuration(
    screen,
    interleaved=False,
    group_q8=False,
    shared_q8=False,
    direct=False,
    model_id=DEFAULT_MODEL,
):
    require_model_variant(model_id, grouped_q8=group_q8 or shared_q8)
    config = json.loads((ROOT / "configs" / SCREENS[screen]).read_text())
    variant = kernel_variant(interleaved, group_q8, shared_q8, direct)
    if variant != "short-mv":
        config["kernel_variant"] = "guarded-single-row" if direct else variant
        if direct:
            config["sampler_variant"] = "direct"
            config["experiment_kind"] = "target_host_sampler"
        config["phase"] = config["phase"].replace(
            "metal_", f"metal_{variant.replace('-', '_')}_", 1
        )
        for condition in config["configurations"]:
            if condition["kernel_profile"] == "short-mv":
                condition["kernel_profile"] = variant
                condition["id"] = condition["id"].replace("short-mv", variant)
    if model_id != DEFAULT_MODEL:
        config["model_id"] = model_id
        config["phase"] = config["phase"].replace(
            "metal_", f"metal_{model_id.replace('-', '_')}_", 1
        )
    return config


def analyze(
    directory,
    *,
    screen="ngram",
    interleaved=False,
    group_q8=False,
    shared_q8=False,
    direct=False,
    model_id=DEFAULT_MODEL,
):
    directory = Path(directory)
    verify(directory)
    frozen = json.loads((directory / "run_manifest.json").read_text())["frozen"]
    expected = screen_configuration(
        screen, interleaved, group_q8, shared_q8, direct, model_id
    )
    candidate = kernel_variant(interleaved, group_q8, shared_q8, direct)
    profile_check = {
        "direct": direct_profile,
        "short-mv": metal_profile,
        "interleaved": interleaved_profile,
        "q8-group": q8_group_profile,
        "q8-shared": q8_shared_profile,
    }[candidate]
    if model_id != DEFAULT_MODEL:
        profile_check = partial(profile_check, model_id=model_id)
    expected_sha = (
        MODEL_SHA256
        if model_id == DEFAULT_MODEL
        else model_identity(model_id)["sha256"]
    )
    corpus = {p["id"]: p for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")}
    if (
        frozen["config"] != expected
        or frozen["target_sha256"] != expected_sha
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
            f"metal_{candidate.replace('-', '_')}_{screen}_comparison_complete"
            if candidate != "short-mv"
            else (
                "metal_short_mv_comparison_complete"
                if screen == "ngram"
                else "metal_retrieval_comparison_complete"
            )
        ),
        "screen": screen,
        "kernel_variant": "guarded-single-row" if direct else candidate,
        "records": len(rows),
        "comparisons": {f"{a}/{b}": paired_summary(table, a, b) for a, b in pairs},
        "warm_comparisons": {f"{a}/{b}": paired_summary(warm, a, b) for a, b in pairs},
        "work_totals": totals,
        "cross_profile_exact": True,
        "same_work_within_each_policy": True,
        "limitation": "Three development prompts, cap128, three repeats. These fit prompts informed suffix threshold/cap exploration. This is a component screen, not held-out or controller evidence.",
    }
    if direct:
        summary["sampler_variant"] = "direct"
        summary["experiment_kind"] = "target_host_sampler"
    if model_id != DEFAULT_MODEL:
        summary["model_id"] = model_id
    atomic_json(directory / "analysis/summary.json", summary)
    return summary
