"""Supplementary controller figures from audited raw measurements, without inference."""

from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from analysis.controller import verify
from analysis.report import cells, paired_summary
from bench.common import atomic_json, sha256


def generate(directory, *, fit=None, out=None):
    directory = Path(directory)
    out = Path(out) if out else directory / "controller-details"
    out.mkdir(parents=True, exist_ok=True)
    rows, manifest = verify(directory)
    cfg = manifest["frozen"]["config"]
    fixed = cfg["selected_fixed"]
    table = cells(rows)
    configs = sorted({r["configuration_id"] for r in rows})
    result = dict(
        schema_version=1,
        synthetic_fixture=manifest["frozen"]["identity"].get(
            "synthetic_fixture", False
        ),
        heldout=cfg["split"] == "heldout",
        source_manifest_sha256=sha256(directory / "run_manifest.json"),
        source_requests_sha256=sha256(directory / "requests.jsonl"),
        selected_fixed=fixed,
        uncertainty="Distinct-prompt bootstrap with both caps together; timing repeats collapsed to medians.",
        categories={},
        policies={},
    )
    references = [r for r in ("target", fixed, "dynamic", "acceptance") if r in configs]
    for category in sorted({r["category"] for r in rows}):
        ids = {r["prompt_id"] for r in rows if r["category"] == category}
        result["categories"][category] = {
            "cost-aware/"
            + ref: paired_summary(table, "cost-aware", ref, filter_ids=ids)
            for ref in references
        }
    for cid in configs:
        rs = [r for r in rows if r["configuration_id"] == cid]
        cycles = sum(r["execution"]["cycles"] for r in rs)
        cost = sum(
            r["execution"]["decision_ms"] + r["execution"]["policy_update_ms"]
            for r in rs
        )
        result["policies"][cid] = dict(
            records=len(rs),
            total_generated_tokens=sum(r["generated_tokens"] for r in rs),
            actual_tokens_range=[
                min(r["generated_tokens"] for r in rs),
                max(r["generated_tokens"] for r in rs),
            ],
            median_ms_per_output_token=statistics.median(
                r["generation_ms"] / r["generated_tokens"] for r in rs
            ),
            median_warm_ms=statistics.median(r["warm_ms"] for r in rs),
            median_first_commit_ms=statistics.median(r["first_token_ms"] for r in rs),
            peak_lifetime_rss_bytes=max(r["peak_rss_bytes"] for r in rs),
            decision_update_ns_per_cycle=cost * 1e6 / cycles,
            hot_policy_fraction=cost / sum(r["generation_ms"] for r in rs),
            mean_policy_init_ms=statistics.mean(r["policy_init_ms"] for r in rs),
            mean_controller_parse_ms=statistics.mean(
                r["controller_parse_ms"] for r in rs
            ),
        )
    result["regressions"] = sorted(
        [
            dict(prompt_id=p, output_cap=cap, ratio=v / table[p, cap, fixed])
            for (p, cap, c), v in table.items()
            if c == "cost-aware"
        ],
        key=lambda r: r["ratio"],
    )
    fit_rows = None
    if fit:
        fit = Path(fit)
        fit_rows, fit_manifest = verify(fit)
        for key in ("binary_sha256", "target_sha256", "draft_sha256", "gpu_layers"):
            if fit_manifest["frozen"][key] != manifest["frozen"][key]:
                raise ValueError("Fit curves use a different operating point")
        result["fit_requests_sha256"] = sha256(fit / "requests.jsonl")
        result["fit_curves"] = {}
        for action in (1, 2, 4, 6, 8):
            rs = [r for r in fit_rows if r["configuration_id"] == f"fixed{action}"]
            if not rs:
                raise ValueError("Missing fixed-cap fit measurements")
            drafted = sum(r["execution"]["drafted"] for r in rs)
            accepted = sum(r["execution"]["accepted"] for r in rs)
            result["fit_curves"][str(action)] = dict(
                acceptance=accepted / drafted if drafted else None,
                proposed=drafted,
                accepted=accepted,
                aggregate_ms_per_output_token=sum(r["generation_ms"] for r in rs)
                / sum(r["generated_tokens"] for r in rs),
                ratio_vs_target=paired_summary(
                    cells(fit_rows), f"fixed{action}", "target"
                ),
            )
    atomic_json(out / "summary.json", result)
    plot(rows, result, table, out)
    return result


def plot(rows, result, table, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = (
        "Synthetic fixtures only; not trained-model evidence"
        if result["synthetic_fixture"]
        else (
            "Frozen held-out measurements"
            if result["heldout"]
            else "Bounded development measurements"
        )
    )
    labels = list(result["policies"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        labels,
        [result["policies"][c]["peak_lifetime_rss_bytes"] / 2**30 for c in labels],
        color="#356b9a",
    )
    ax.tick_params(axis="x", rotation=35)
    ax.set(
        ylabel="Lifetime process peak RSS (GiB)",
        title=title + "\nIncludes loading and warm-up; not policy-only memory",
    )
    fig.tight_layout()
    fig.savefig(out / "memory.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for category, comparisons in result["categories"].items():
        ids = {r["prompt_id"] for r in rows if r["category"] == category}
        ratios = [
            v / table[p, cap, result["selected_fixed"]]
            for (p, cap, c), v in table.items()
            if c == "cost-aware" and p in ids
        ]
        axes[0].ecdf(ratios, label=f"{category}: {len(ids)} prompts")
    axes[0].axvline(1, color="gray", linestyle="--")
    axes[0].set(
        xlabel="Cost-aware / frozen fixed throughput (cell medians)",
        ylabel="Fraction of prompt/cap cells",
    )
    axes[0].legend(fontsize=8)
    candidates = [
        c
        for c in (
            "target",
            "dynamic",
            "acceptance",
            "cost-aware",
            "no-fallback",
            "no-probes",
        )
        if c in labels
    ]
    for i, cid in enumerate(candidates):
        metric = paired_summary(table, cid, result["selected_fixed"])
        axes[1].errorbar(
            metric["ratio"],
            i,
            xerr=[
                [metric["ratio"] - metric["ci95"][0]],
                [metric["ci95"][1] - metric["ratio"]],
            ],
            fmt="o",
        )
    axes[1].set_yticks(range(len(candidates)), candidates)
    axes[1].axvline(1, color="gray", linestyle="--")
    axes[1].set_xlabel("Throughput / frozen fixed; prompt-bootstrap 95% interval")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "ablations-and-regressions.png", dpi=160)
    plt.close(fig)
    if "fit_curves" in result:
        curves = result["fit_curves"]
        caps = [int(a) for a in curves]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(
            caps,
            [curves[str(a)]["aggregate_ms_per_output_token"] for a in caps],
            marker="o",
        )
        axes[0].set(
            xlabel="Fixed requested cap", ylabel="Full generation ms / output token"
        )
        axes[1].plot(caps, [curves[str(a)]["acceptance"] for a in caps], marker="o")
        axes[1].set(
            xlabel="Fixed requested cap",
            ylabel="Accepted / actual proposed drafts",
            ylim=(0, 1.05),
        )
        fig.suptitle(title + " — calibration fit data only")
        fig.tight_layout()
        fig.savefig(out / "fit-cost-acceptance.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    generate(args.run, fit=args.fit, out=args.out)
