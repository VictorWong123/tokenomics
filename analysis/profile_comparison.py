"""Paired CPU-profile and logging comparison; separate from controller feasibility."""

import argparse
import json
import math
from pathlib import Path
import statistics

from bench.common import atomic_json, read_jsonl
from analysis.report import cells, paired_summary, validate_run, draw_interval

PAIRS = (
    ("packed-target", "target"),
    ("packed-fixed8", "guarded-fixed8"),
    ("packed-fixed8", "packed-target"),
    ("guarded-fixed8", "target"),
    ("guarded-detailed", "target"),
    ("packed-detailed", "packed-target"),
)


def analyze(directory):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    rows = read_jsonl(directory / "requests.jsonl")
    validate_run(rows, manifest, json.loads((directory / "complete.json").read_text()))
    frozen = manifest["frozen"]
    cfg = frozen["config"]
    if (
        cfg.get("phase") != "cpu_profile_comparison"
        or len(frozen["prompts"]) != 3
        or cfg["repetitions"] != 3
        or cfg["output_caps"] != [128]
    ):
        raise ValueError(
            "Profile study requires the complete three-prompt repeated grid"
        )
    configs = {c["id"]: c for c in cfg["configurations"]}
    expected = {c for pair in PAIRS for c in pair}
    if (
        set(configs) != expected
        or set(frozen["configuration_binary_sha256"]) != expected
    ):
        raise ValueError("Incomplete paired build identity")
    for cid, conf in configs.items():
        profile = "packed" if cid.startswith("packed-") else "guarded"
        if conf["kernel_profile"] != profile or conf["detailed"] != cid.endswith(
            "detailed"
        ):
            raise ValueError("Unexpected profile/logging configuration")
    for profile in ("guarded", "packed"):
        digests = {
            frozen["configuration_binary_sha256"][cid]
            for cid, c in configs.items()
            if c["kernel_profile"] == profile
        }
        if len(digests) != 1:
            raise ValueError("One profile used different binaries across conditions")
    checks = read_jsonl(directory / "correctness.jsonl")
    if len(checks) != len(rows) or not all(c["ok"] for c in checks):
        raise ValueError("Non-passing profile comparison")
    groups = {}
    for row in rows:
        conf = configs[row["configuration_id"]]
        marker = (
            "TOKENOMICS_CPU_PACKED_VERIFY=ON"
            if conf["kernel_profile"] == "packed"
            else "TOKENOMICS_CPU_BATCH_INVARIANT=ON"
        )
        if (
            marker not in row["system_info"]
            or row["settings"]["detailed"] != conf["detailed"]
            or any(
                row["settings"][k] != conf[k]
                for k in ("policy", "k", "drafter")
                if k in conf
            )
            or any(
                row["settings"][k] != cfg[k] for k in ("context", "batch", "threads")
            )
            or row["settings"]["gpu_layers"] != 0
        ):
            raise ValueError(
                "Profile/instrumentation execution differs from configuration"
            )
        if row["generated_tokens"] != len(row["token_ids"]):
            raise ValueError("Generated-token count mismatch")
        if row["peak_rss_bytes"] <= 0:
            raise ValueError("Missing measured memory")
        cycles = read_jsonl(directory / row["source_cycles"])
        measured = [c for c in cycles if not c.get("warmup")]
        if conf["detailed"]:
            if sum(c["committed"] for c in measured) != row["generated_tokens"]:
                raise ValueError("Detailed-cycle accounting mismatch")
        elif cycles:
            raise ValueError("Minimal configuration retained cycle records")
        key = row["prompt_id"], row["output_cap"], row["repetition"]
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        ref = next(r for r in group if r["configuration_id"] == "target")
        if any(
            r["token_ids"] != ref["token_ids"]
            or r["prompt_ids"] != ref["prompt_ids"]
            or r["stop"] != ref["stop"]
            for r in group
        ):
            raise ValueError("Cross-profile output mismatch")
    table = cells(rows)
    comparisons = {f"{c}/{r}": paired_summary(table, c, r) for c, r in PAIRS}
    overhead = {}
    for profile, baseline in (("guarded", "target"), ("packed", "packed-target")):
        metric = comparisons[f"{profile}-detailed/{baseline}"]
        overhead[profile] = {
            "decode_cost_fraction": 1 / metric["ratio"] - 1,
            "ci95": [1 / metric["ci95"][1] - 1, 1 / metric["ci95"][0] - 1],
            "boundary": "Cycle retention/bookkeeping; JSONL serialization is outside decode, inside process latency.",
        }
    result = {
        "schema_version": 1,
        "phase": cfg["phase"],
        "synthetic_fixture": frozen["identity"]["synthetic_fixture"],
        "full_feasibility_cleared": False,
        "comparisons": comparisons,
        "logging_overhead": overhead,
        "baseline_table": {
            cid: {
                "geomean_median_tps": math.exp(
                    statistics.mean(
                        math.log(v) for (_, _, c), v in table.items() if c == cid
                    )
                ),
                "peak_lifetime_rss_bytes": max(
                    r["peak_rss_bytes"] for r in rows if r["configuration_id"] == cid
                ),
                "median_process_ms": statistics.median(
                    r["process_ms"] for r in rows if r["configuration_id"] == cid
                ),
            }
            for cid in configs
        },
        "limitation": "Paired component study on three development fit prompts, one output cap and three repeats. No full Pilot A/B, predictive scheduling, held-out or Mac result. Negative logging-overhead estimates reflect timing noise.",
    }
    out = directory / "analysis"
    atomic_json(out / "summary.json", result)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(comparisons)
    values = [comparisons[k]["ratio"] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, values)
    for i, label in enumerate(labels):
        metric = comparisons[label]
        draw_interval(ax, metric["ratio"], i, metric["ci95"], color="black")
    ax.axvline(1, color="black", linestyle="--")
    ax.set(
        xlabel="Paired throughput ratio (prompt-clustered 95% interval)",
        title="CPU profile / logging component study — development only",
    )
    fig.tight_layout()
    fig.savefig(out / "profiles.png", dpi=160)
    plt.close(fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(analyze(parser.parse_args().directory), indent=2))
