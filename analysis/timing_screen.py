"""Describe a timing screen without treating it as repeated feasibility evidence."""

import argparse
import json
import math
from pathlib import Path
from bench.common import atomic_json, read_jsonl
from analysis.report import validate_run
from bench.run import first_mismatch


def report(run, out):
    rows = read_jsonl(run / "requests.jsonl")
    manifest = json.loads((run / "run_manifest.json").read_text())
    complete = json.loads((run / "complete.json").read_text())
    validate_run(rows, manifest, complete)
    cfg = manifest["frozen"]["config"]
    if (
        cfg.get("split") != "development"
        or cfg["repetitions"] != 1
        or len(cfg["output_caps"]) != 1
    ):
        raise ValueError("screen analysis requires one development cap and repetition")
    checks = read_jsonl(run / "correctness.jsonl")
    if len(checks) != len(rows) or any(c.get("ok") is not True for c in checks):
        raise ValueError("non-passing screen correctness")
    grouped = {}
    for row in rows:
        key = row["prompt_id"], row["output_cap"], row["repetition"]
        grouped.setdefault(key, {})[row["configuration_id"]] = row
    ratios, costs = {}, []
    for group in grouped.values():
        ref = group["target"]
        for cid, row in group.items():
            if (
                row.get("correctness") != "pass"
                or first_mismatch(ref["token_ids"], row["token_ids"]) is not None
                or ref["stop"] != row["stop"]
            ):
                raise ValueError("screen output divergence")
            ratio = (
                row["generated_tokens"]
                / row["generation_ms"]
                / (ref["generated_tokens"] / ref["generation_ms"])
            )
            ratios.setdefault(cid, []).append(ratio)
            cs = [
                c
                for c in read_jsonl(run / row["source_cycles"])
                if not c.get("warmup", False)
            ]
            if sum(c["committed"] for c in cs) != row["generated_tokens"]:
                raise ValueError("screen cycle accounting mismatch")
            drafted = sum(c["actual_drafts"] for c in cs)
            accepted = sum(c["accepted"] for c in cs)
            costs.append(
                {
                    "prompt_id": row["prompt_id"],
                    "configuration_id": cid,
                    "cycles": len(cs),
                    "drafted": drafted,
                    "accepted": accepted,
                    "acceptance": accepted / drafted if drafted else None,
                    **{
                        k: sum(c[k] for c in cs)
                        for k in (
                            "draft_ms",
                            "verify_ms",
                            "sync_ms",
                            "decision_ms",
                            "cache_ms",
                            "wall_ms",
                        )
                    },
                }
            )
    summary = {
        "schema_version": 1,
        "status": "development timing screen only",
        "source": manifest["frozen"]["code"],
        "prompts": len(manifest["frozen"]["prompts"]),
        "repetitions": manifest["frozen"]["config"]["repetitions"],
        "output_caps": manifest["frozen"]["config"]["output_caps"],
        "geomean_ratios_vs_target": {
            c: math.exp(sum(map(math.log, v)) / len(v)) for c, v in ratios.items()
        },
        "q1_supported": None,
        "q2_supported": None,
        "synthetic_fixture": manifest["frozen"]
        .get("identity", {})
        .get("synthetic_fixture", False),
        "limitation": "No repeated-pilot decision or controller gain follows from this screen.",
    }
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "measurements.json", summary)
    atomic_json(out / "stage-costs.json", costs)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(ratios)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, [summary["geomean_ratios_vs_target"][c] for c in labels])
    ax.axhline(1, color="black", linestyle="--")
    ax.set(
        ylabel="Geometric mean throughput ratio to target-only",
        title=f"Development timing screen: {summary['prompts']} prompts, "
        f"{summary['repetitions']} repetition(s)\nPreliminary; no feasibility conclusion",
    )
    fig.tight_layout()
    fig.savefig(out / "timing.png", dpi=160)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.out)
