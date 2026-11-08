"""Paired, prompt-clustered summaries. Failed or incomplete runs cannot enter."""

from __future__ import annotations
import argparse
import collections
import json
import math
import pathlib
import statistics
import numpy as np
from bench.common import atomic_json, read_jsonl


def draw_interval(ax, value, position, interval, *, color="#356b9a"):
    """Draw percentile bounds directly: they need not bracket the point exactly.

    NumPy/bootstrap and scalar reductions can differ by one ULP for a
    degenerate distribution; translating bounds into signed xerr can fail.
    Never clamp or change the reported statistics to make a plot render.
    """
    low, high = interval
    if not all(math.isfinite(v) for v in (value, low, high)) or low > high:
        raise ValueError("Invalid plotted interval")
    ax.hlines(position, low, high, color=color, linewidth=1.5)
    ax.plot(value, position, "o", color=color)


def diagnostic_request(row):
    settings = row.get("settings", {})
    return any(
        settings.get(k)
        for k in (
            "diagnostic",
            "diagnostic_logits",
            "max_cycles",
            "capture_checkpoints",
            "resume_checkpoint",
        )
    )


def cells(rows):
    grouped = collections.defaultdict(list)
    for r in rows:
        if r.get("correctness") != "pass":
            raise ValueError("non-passing correctness record")
        if diagnostic_request(r):
            raise ValueError("diagnostic requests cannot enter performance summaries")
        if (
            not math.isfinite(r["generation_ms"])
            or r["generation_ms"] <= 0
            or r["generated_tokens"] <= 0
        ):
            raise ValueError("no measurable generated tokens")
        grouped[(r["prompt_id"], r["output_cap"], r["configuration_id"])].append(
            r["generated_tokens"] * 1000 / r["generation_ms"]
        )
    return {k: statistics.median(v) for k, v in grouped.items()}


def paired_summary(
    table, challenger, reference, seed=20261107, samples=10000, filter_ids=None
):
    by_prompt = collections.defaultdict(list)
    for (pid, cap, cid), v in table.items():
        if cid != challenger or (filter_ids is not None and pid not in filter_ids):
            continue
        if (pid, cap, reference) not in table:
            raise ValueError("unpaired comparison")
        by_prompt[pid].append(math.log(v / table[(pid, cap, reference)]))
    if not by_prompt:
        raise ValueError("empty paired comparison")
    # Each prompt's caps stay together. Repeats have already been collapsed.
    prompt_logs = np.array([statistics.mean(by_prompt[k]) for k in sorted(by_prompt)])
    rng = np.random.default_rng(seed)
    draws = np.exp(
        rng.choice(prompt_logs, (samples, len(prompt_logs)), replace=True).mean(axis=1)
    )
    ratios = [math.exp(v) for values in by_prompt.values() for v in values]
    return {
        "ratio": math.exp(float(prompt_logs.mean())),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "prompts": len(by_prompt),
        "prompt_cap_cells": len(ratios),
        "wins": sum(x > 1.05 for x in ratios),
        "ties": sum(1 / 1.05 <= x <= 1.05 for x in ratios),
        "losses": sum(x < 1 / 1.05 for x in ratios),
        "more_than_10pct_slower": sum(x < 1 / 1.10 for x in ratios),
        "worst_ratio": min(ratios),
    }


def select_fixed(table, ids=None):
    actions = sorted({k[2] for k in table if k[2].startswith("fixed")})
    if not actions:
        raise ValueError("no fixed actions")
    return max(
        actions,
        key=lambda c: statistics.mean(
            math.log(v)
            for (p, _, a), v in table.items()
            if a == c and (ids is None or p in ids)
        ),
    )


def summarize(rows, samples=10000, frozen_fixed=None):
    table = cells(rows)
    configs = sorted({k[2] for k in table})
    fixed = frozen_fixed if frozen_fixed is not None else select_fixed(table)
    if fixed not in configs or not fixed.startswith("fixed"):
        raise ValueError("frozen fixed baseline is absent or invalid")
    output = {
        "schema_version": 1,
        "selected_fixed": fixed,
        "selection_rule": (
            "fixed baseline frozen on separate development data"
            if frozen_fixed is not None
            else "maximum geometric mean prompt/cap median decode throughput on development only"
        ),
        "comparisons": {},
        "baseline_table": {},
        "categories": {},
    }
    for cid in configs:
        vals = [v for (_, _, c), v in table.items() if c == cid]
        rs = [r for r in rows if r["configuration_id"] == cid]
        output["baseline_table"][cid] = {
            "geomean_median_tps": math.exp(statistics.mean(map(math.log, vals))),
            "median_warm_ms": statistics.median(r["warm_ms"] for r in rs),
            "median_load_ms": statistics.median(r["load_ms"] for r in rs),
            "peak_lifetime_rss_bytes": max(r["peak_rss_bytes"] for r in rs),
            "cells": len(vals),
        }
        for ref in dict.fromkeys(["target", fixed, "confidence"]):
            if ref in configs:
                output["comparisons"][f"{cid}/{ref}"] = paired_summary(
                    table, cid, ref, samples=samples
                )
    for category in sorted({r["category"] for r in rows}):
        ids = {r["prompt_id"] for r in rows if r["category"] == category}
        output["categories"][category] = paired_summary(
            table, fixed, "target", samples=samples, filter_ids=ids
        )
    # Fit/validation split is by distinct prompt, never timing repeat.
    ids = sorted({r["prompt_id"] for r in rows})
    fit = {p for p in ids if int(p.rsplit("-", 1)[1]) % 6 < 3}
    validation = set(ids) - fit
    if frozen_fixed is None and fit and validation:
        fit_fixed = select_fixed(table, fit)
        output["q1_validation"] = {
            "fit_selected_fixed": fit_fixed,
            **paired_summary(
                table, fit_fixed, "target", samples=samples, filter_ids=validation
            ),
        }
        q = output["q1_validation"]
        output["q1_supported"] = q["ratio"] > 1.05 and q["ci95"][0] > 1.0
    else:
        output["q1_supported"] = False
    return output


def validate_run(rows, manifest, complete):
    """Require the exact frozen Cartesian product before estimating performance."""
    frozen = manifest["frozen"]
    cfg = frozen["config"]
    if any(diagnostic_request(row) for row in rows):
        raise ValueError("diagnostic requests cannot enter performance summaries")
    expected = {
        (p["id"], cap, c["id"], rep)
        for p in frozen["prompts"]
        for cap in cfg["output_caps"]
        for c in cfg["configurations"]
        for rep in range(cfg["repetitions"])
    }
    actual = [
        (r["prompt_id"], r["output_cap"], r["configuration_id"], r["repetition"])
        for r in rows
    ]
    if (
        not expected
        or len(actual) != len(set(actual))
        or set(actual) != expected
        or complete.get("records") != len(actual)
    ):
        raise ValueError("incomplete, duplicate or unexpected frozen measurement cells")
    for conf in cfg["configurations"]:
        if conf.get("policy") == "cost-aware":
            from bench.controller import point_from_frozen, validate_binding

            prior = conf.get("controller", {})
            fixture = frozen.get("identity", {}).get("synthetic_fixture", False)
            point = point_from_frozen(
                frozen,
                conf.get("drafter", "model"),
                conf.get("cache", "retain"),
                configuration_id=conf["id"],
            )
            validate_binding(prior, point, synthetic_fixture=fixture)
            if not fixture and {p["id"] for p in frozen["prompts"]} & set(
                prior["provenance"]["fit_prompt_ids"]
            ):
                raise ValueError(
                    "Controller validation overlaps development calibration"
                )
            for row in rows:
                if row["configuration_id"] == conf["id"]:
                    validate_binding(
                        prior,
                        point,
                        synthetic_fixture=fixture,
                        system_info=row["system_info"],
                    )
                    if not row.get("controller_active"):
                        raise ValueError(
                            "Controller measurement did not execute the controller"
                        )
    if cfg.get("split") == "heldout":
        if not cfg.get("selected_fixed") or not cfg.get("development_selection_sha256"):
            raise ValueError(
                "held-out analysis requires a frozen development baseline selection"
            )
        if any(p.get("split") != "heldout" for p in frozen["prompts"]):
            raise ValueError("mixed development/evaluation prompts")
        return cfg["selected_fixed"]
    if any(p.get("split", "development") != "development" for p in frozen["prompts"]):
        raise ValueError("evaluation data cannot select a development baseline")
    return None


def write_report(run_dir, out):
    run_dir = pathlib.Path(run_dir)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "complete.json").exists():
        raise ValueError("run incomplete: no performance report")
    rows = read_jsonl(run_dir / "requests.jsonl")
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    complete = json.loads((run_dir / "complete.json").read_text())
    frozen_fixed = validate_run(rows, manifest, complete)
    result = summarize(rows, frozen_fixed=frozen_fixed)
    result["phase"] = manifest["frozen"]["config"].get("phase", "pilot")
    result["hardware"] = manifest["hardware_start"]
    result["synthetic_fixture"] = (
        manifest["frozen"].get("identity", {}).get("synthetic_fixture", False)
    )
    if result["synthetic_fixture"]:
        result["q1_supported"] = False
    atomic_json(out / "summary.json", result)
    title = (
        "Synthetic fixture pipeline demonstration — NOT feasibility evidence"
        if result["synthetic_fixture"]
        else "Tokenomics platform-local pilot"
    )
    lines = [
        f"# {title}",
        "",
        f"Selected development fixed setting: `{result['selected_fixed']}`.",
        "",
        "| Configuration | Decode tokens/s (geometric mean of cell medians) | Median warm latency ms | Peak lifetime RSS MiB |",
        "|---|---:|---:|---:|",
    ]
    for cid, v in result["baseline_table"].items():
        lines.append(
            f"| {cid} | {v['geomean_median_tps']:.2f} | {v['median_warm_ms']:.2f} | {v['peak_lifetime_rss_bytes']/2**20:.1f} |"
        )
    lines += [
        "",
        "Paired uncertainty resamples distinct prompts with both output caps together. Timing repetitions are not independent prompts. Model load, request setup, prompt processing and complete generation are retained separately. No memory, energy or dollar-cost reduction is inferred from throughput.",
        "",
        "Q1 validation: "
        + json.dumps(result.get("q1_validation", {"status": "insufficient data"})),
        "",
        "Q2 is a separate checkpoint replay investigation; this table does not establish predictable within-generation adaptation.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    plot(rows, result, out, run_dir)
    return result


def plot(rows, summary, out, source_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = cells(rows)
    fixed = summary["selected_fixed"]
    labels = list(summary["baseline_table"])
    subtitle = (
        "Synthetic fixtures — pipeline check only"
        if summary.get("synthetic_fixture")
        else "Platform-local measurements"
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, [summary["baseline_table"][c]["geomean_median_tps"] for c in labels])
    ax.set_ylabel("Committed tokens / generation second")
    ax.set_title(subtitle)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out / "throughput.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for category in sorted({r["category"] for r in rows}):
        ids = {r["prompt_id"] for r in rows if r["category"] == category}
        v = [
            x / t[(p, cap, "target")]
            for (p, cap, c), x in t.items()
            if c == fixed and p in ids
        ]
        ax.ecdf(v, label=category)
    ax.axvline(1, color="black", linestyle="--")
    ax.set(
        xlabel=f"{fixed} / target-only throughput",
        ylabel="Fraction of prompt/cap cells",
        title=subtitle,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "paired-ratios.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        labels,
        [
            summary["baseline_table"][c]["peak_lifetime_rss_bytes"] / 2**20
            for c in labels
        ],
    )
    ax.set_ylabel("Peak lifetime RSS MiB (includes warm-up)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out / "memory.png")
    plt.close(fig)
    # Diagnostic fixed-cap acceptance/cost and representative action trajectories.
    run_dir = pathlib.Path(source_dir) if source_dir else None
    all_cycles = []
    if run_dir:
        seen_paths = set()
        for r in rows:
            p = run_dir / r["source_cycles"]
            if p.exists() and p not in seen_paths:
                seen_paths.add(p)
                all_cycles.extend(
                    [c for c in read_jsonl(p) if not c.get("warmup", False)]
                )
    if all_cycles:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for cid in [c for c in labels if c.startswith("fixed")]:
            cs = [c for c in all_cycles if c["configuration_id"] == cid]
            drafted = sum(c["actual_drafts"] for c in cs)
            axes[0].scatter(
                int(cid[5:]), sum(c["accepted"] for c in cs) / drafted if drafted else 0
            )
            axes[1].scatter(
                int(cid[5:]),
                sum(c["wall_ms"] for c in cs) / sum(c["committed"] for c in cs),
            )
        axes[0].set(xlabel="Draft cap", ylabel="Accepted / actual drafts")
        axes[1].set(xlabel="Draft cap", ylabel="Complete cycle ms / committed token")
        fig.tight_layout()
        fig.savefig(out / "fixed-cost-acceptance.png")
        plt.close(fig)
        ratios = sorted(
            (v / t[(p, cap, "target")], p, cap)
            for (p, cap, c), v in t.items()
            if c == fixed
        )
        selected = [
            ("worst regression", ratios[0]),
            ("median ratio", ratios[len(ratios) // 2]),
        ]
        fig, axes = plt.subplots(2, 1, figsize=(9, 6))
        for ax, (label, (_, pid, cap)) in zip(axes, selected):
            cs = [
                c
                for c in all_cycles
                if c["prompt_id"] == pid
                and c["output_cap"] == cap
                and c["configuration_id"] in ("confidence", "acceptance")
                and c["repetition"] == 0
            ]
            for cid in ("confidence", "acceptance"):
                cc = [c for c in cs if c["configuration_id"] == cid]
                ax.step(
                    [c["cycle"] for c in cc],
                    [c["action"] for c in cc],
                    where="post",
                    label=cid,
                )
            ax.set(
                title=f"{label}: {pid}, cap={cap}",
                xlabel="Cycle",
                ylabel="Requested cap",
            )
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "action-trajectories.png")
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path)
    a = p.parse_args()
    write_report(a.run, a.out or a.run / "analysis")
