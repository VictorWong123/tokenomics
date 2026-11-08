"""Audit and summarize a frozen controller development comparison."""

from __future__ import annotations
import json
import math
import statistics
from pathlib import Path
from analysis.report import cells, paired_summary, summarize, validate_run
from bench.common import atomic_json, read_jsonl, sha256
from bench.controller import point_from_frozen, validate_binding


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    complete = json.loads((directory / "complete.json").read_text())
    rows = read_jsonl(directory / "requests.jsonl")
    validate_run(rows, manifest, complete)
    frozen = manifest["frozen"]
    fixture = frozen.get("identity", {}).get("synthetic_fixture", False)
    configurations = {c["id"]: c for c in frozen["config"]["configurations"]}
    retained = []
    receipts = list((directory / "blocks").glob("*/complete.json"))
    if len(receipts) != complete["blocks"]:
        raise ValueError("Missing immutable comparison blocks")
    for path in receipts:
        receipt = json.loads(path.read_text())
        if not all(c.get("ok") is True for c in receipt["correctness"]):
            raise ValueError("Non-passing controller correctness")
        for name in receipt["records"]:
            record = path.parent / name
            if sha256(record) != receipt["sha256"][name]:
                raise ValueError("Immutable controller record hash changed")
            retained.append(json.loads(record.read_text()))
    key = lambda r: (
        r["prompt_id"],
        r["output_cap"],
        r["repetition"],
        r["configuration_id"],
    )
    if sorted(rows, key=key) != sorted(retained, key=key):
        raise ValueError("Aggregate controller records differ from immutable blocks")
    groups = {}
    for row in rows:
        groups.setdefault(key(row)[:3], {})[row["configuration_id"]] = row
    for group in groups.values():
        ref = group["target"]
        for cid, row in group.items():
            conf = configurations[cid]
            if (
                row.get("correctness") != "pass"
                or row["token_ids"] != ref["token_ids"]
                or row["stop"] != ref["stop"]
                or row["prompt_ids"] != ref["prompt_ids"]
                or row["generated_tokens"] != len(row["token_ids"])
            ):
                raise ValueError("Controller output divergence")
            raw = [
                r
                for r in read_jsonl(directory / row["source_record"])
                if not r["warmup"]
            ]
            if len(raw) != 1 or any(row.get(k) != v for k, v in raw[0].items()):
                raise ValueError("Raw controller record differs from measurement")
            for setting in (
                "policy",
                "k",
                "cache",
                "drafter",
                "confidence",
                "detailed",
            ):
                if setting in conf and row["settings"][setting] != conf[setting]:
                    raise ValueError("Controller condition setting changed")
            if any(
                row["settings"][k] != frozen["config"][k]
                for k in ("context", "batch", "threads")
            ):
                raise ValueError("Controller execution settings changed")
            expected_cache = len(row["prompt_ids"]) + row["generated_tokens"] - 1
            if row["generated_tokens"] and (
                row["target_cached"] != expected_cache
                or not 0 <= row["draft_cached"] <= expected_cache
            ):
                raise ValueError("Controller final cache accounting differs")
            if (
                row["peak_rss_bytes"] <= 0
                or row["generated_tokens"] > row["output_cap"]
                or row["settings"]["max_tokens"] != row["output_cap"]
                or any(
                    not math.isfinite(row[k]) or row[k] < 0
                    for k in (
                        "generation_ms",
                        "warm_ms",
                        "policy_init_ms",
                        "controller_parse_ms",
                    )
                )
                or row["generation_ms"] <= 0
                or row["warm_ms"] < row["generation_ms"]
            ):
                raise ValueError("Invalid controller timing/resource measurement")
            totals = row["execution"]
            if (
                sum(a["committed_tokens"] for a in totals["actions"])
                != row["generated_tokens"]
                or sum(a["cycles"] for a in totals["actions"]) != totals["cycles"]
                or totals["verified_rows"] != totals["cycles"] + totals["drafted"]
                or totals["accepted"] > totals["drafted"]
            ):
                raise ValueError("Controller request accounting mismatch")
            cycles = [
                c
                for c in read_jsonl(directory / row["source_cycles"])
                if not c["warmup"]
            ]
            detailed = conf.get("detailed", frozen["config"].get("detailed", True))
            if detailed:
                if (
                    len(cycles) != totals["cycles"]
                    or sum(c["committed"] for c in cycles) != row["generated_tokens"]
                ):
                    raise ValueError("Controller detailed cycle accounting mismatch")
            elif cycles:
                raise ValueError(
                    "Minimal request unexpectedly retained detailed cycles"
                )
            if conf["policy"] == "cost-aware":
                prior = conf["controller"]
                if (
                    not fixture
                    and row["prompt_id"] in prior["provenance"]["fit_prompt_ids"]
                ):
                    raise ValueError("Controller validation overlaps calibration fit")
                validate_binding(
                    prior,
                    point_from_frozen(
                        frozen,
                        conf.get("drafter", "model"),
                        conf.get("cache", "retain"),
                    ),
                    synthetic_fixture=fixture,
                    system_info=row["system_info"],
                )
                if (
                    row["settings"]["controller"] != prior
                    or not row["controller_active"]
                ):
                    raise ValueError(
                        "Executed controller differs from frozen calibration"
                    )
                if detailed and any(
                    c.get("controller", {}).get("selected") != c["action"]
                    for c in cycles
                ):
                    raise ValueError("Missing controller decision trace")
    return rows, manifest


def report(directory, out=None, *, plots=True):
    directory = Path(directory)
    out = Path(out) if out else directory / "analysis"
    rows, manifest = verify(directory)
    cfg = manifest["frozen"]["config"]
    fixed = cfg["selected_fixed"]
    result = summarize(rows, frozen_fixed=fixed)
    table = cells(rows)
    configs = sorted({r["configuration_id"] for r in rows})
    for cid in configs:
        for ref in ("dynamic", "acceptance", "resident", "cost-aware"):
            if ref in configs:
                result["comparisons"][f"{cid}/{ref}"] = paired_summary(table, cid, ref)
    costs = {}
    for cid in configs:
        rs = [r for r in rows if r["configuration_id"] == cid]
        gen = sum(r["generation_ms"] for r in rs)
        stages = {
            k: sum(r["execution"][k] for r in rs)
            for k in (
                "cycles",
                "drafted",
                "accepted",
                "verified_rows",
                "synced_tokens",
                "draft_replay_tokens",
                "decision_ms",
                "policy_update_ms",
                "sync_ms",
                "draft_ms",
                "verify_ms",
                "cache_ms",
                "cycle_wall_ms",
            )
        }
        stages.update(
            generation_ms=gen,
            policy_init_ms=sum(r["policy_init_ms"] for r in rs),
            controller_parse_ms=sum(r["controller_parse_ms"] for r in rs),
            hot_policy_fraction=(stages["decision_ms"] + stages["policy_update_ms"])
            / gen,
            unassigned_generation_ms=gen - stages["cycle_wall_ms"],
            acceptance=(
                stages["accepted"] / stages["drafted"] if stages["drafted"] else None
            ),
        )
        if cid in ("cost-aware", "no-fallback", "no-probes", "cost-aware-detailed"):
            stages["controller_totals"] = {
                k: sum(r["controller"][k] for r in rs) for k in rs[0]["controller"]
            }
            stages["mean_prediction_absolute_fraction"] = (
                stages["controller_totals"]["prediction_abs_fraction"]
                / stages["cycles"]
            )
        stages["actions"] = [
            dict(
                action=a["action"],
                **{
                    k: sum(r["execution"]["actions"][i][k] for r in rs)
                    for k in ("cycles", "committed_tokens", "wall_ms")
                },
            )
            for i, a in enumerate(rs[0]["execution"]["actions"])
        ]
        costs[cid] = stages
    result.pop("q1_supported", None)
    result.update(
        phase=cfg["phase"],
        synthetic_fixture=manifest["frozen"]
        .get("identity", {})
        .get("synthetic_fixture", False),
        full_feasibility_cleared=False,
        heldout=False,
        limitation="Bounded, frozen development engineering comparison; uncertainty resamples distinct prompts, not repetitions. No full Pilot A/B or 90-prompt held-out result.",
        stage_costs=costs,
    )
    if "cost-aware" in configs:
        ratios = sorted(
            (
                statistics.mean(
                    math.log(table[p, cap, "cost-aware"] / table[p, cap, "target"])
                    for pp, cap, c in table
                    if pp == p and c == "target"
                ),
                p,
            )
            for p in sorted({r["prompt_id"] for r in rows})
        )
        result["trace_prompt_selection"] = {
            "worst_regression": ratios[0][1],
            "median_improvement": ratios[len(ratios) // 2][1],
        }
    atomic_json(out / "summary.json", result)
    lines = [
        "# Controller development measurements",
        "",
        result["limitation"],
        "",
        f"Frozen fixed comparator: `{fixed}`. Synthetic fixtures: `{result['synthetic_fixture']}`.",
        "",
        "| Policy | Decode tokens/s¹ | Median warm ms | Peak RSS MiB | Decision + update / generation |",
        "|---|---:|---:|---:|---:|",
    ]
    for cid, value in result["baseline_table"].items():
        lines.append(
            f"| {cid} | {value['geomean_median_tps']:.3f} | {value['median_warm_ms']:.2f} | {value['peak_lifetime_rss_bytes']/2**20:.1f} | {costs[cid]['hot_policy_fraction']:.3%} |"
        )
    lines += [
        "",
        "¹ Geometric mean of prompt/cap median throughput. Full probe, synchronization and update work remains charged. RSS is lifetime process peak, not policy memory savings.",
        "",
        "| Comparison | Throughput ratio | Prompt-bootstrap 95% interval |",
        "|---|---:|---:|",
    ]
    for name, value in result["comparisons"].items():
        if name.startswith("cost-aware/"):
            lines.append(
                f"| {name} | {value['ratio']:.4f} | [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}] |"
            )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    if plots:
        plot(result, out)
    return result


def plot(result, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(result["baseline_table"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, cid in enumerate(labels):
        metric = result["comparisons"][cid + "/target"]
        axes[0].errorbar(
            metric["ratio"],
            i,
            xerr=[
                [metric["ratio"] - metric["ci95"][0]],
                [metric["ci95"][1] - metric["ratio"]],
            ],
            fmt="o",
        )
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].axvline(1, color="gray", linestyle="--")
    axes[0].set_xlabel("Decode throughput / ordinary target (prompt bootstrap)")
    bottom = [0.0] * len(labels)
    for stage in (
        "verify_ms",
        "draft_ms",
        "sync_ms",
        "cache_ms",
        "decision_ms",
        "policy_update_ms",
    ):
        values = [
            result["stage_costs"][c][stage] / result["stage_costs"][c]["generation_ms"]
            for c in labels
        ]
        axes[1].barh(labels, values, left=bottom, label=stage.removesuffix("_ms"))
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[1].set_xlabel("Measured stage time / complete generation time")
    axes[1].legend(fontsize=7)
    fig.suptitle(
        "Synthetic fixtures only"
        if result["synthetic_fixture"]
        else "Bounded development controller study"
    )
    fig.tight_layout()
    fig.savefig(out / "controller-comparison.png", dpi=160)
    plt.close(fig)


def plot_traces(directory):
    """Fresh diagnostic traces chosen by the frozen median/worst prompt rule."""
    directory = Path(directory)
    rows, _ = verify(directory)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for row in rows:
        if row["configuration_id"] != "cost-aware-detailed" or row["repetition"] != 0:
            continue
        cycles = [
            c for c in read_jsonl(directory / row["source_cycles"]) if not c["warmup"]
        ]
        index = [c["cycle"] for c in cycles]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6))
        axes[0, 0].step(
            index, [c["action"] for c in cycles], where="mid", label="selected cap"
        )
        probes = [c for c in cycles if c["controller"]["probe"]]
        axes[0, 0].scatter(
            [c["cycle"] for c in probes],
            [c["action"] for c in probes],
            marker="x",
            color="red",
            label="probe",
        )
        axes[0, 0].set_ylabel("Requested action (0 = target only)")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].plot(index, [1000 * c["committed"] / c["wall_ms"] for c in cycles])
        axes[0, 1].set_ylabel("Committed tokens / cycle second")
        axes[1, 0].plot(
            index, [c["controller"]["predicted_ms"] for c in cycles], label="prediction"
        )
        axes[1, 0].plot(
            index, [c["observation_ms"] for c in cycles], label="actual before update"
        )
        axes[1, 0].set_ylabel("Complete observed cycle ms")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].plot(
            index,
            [
                c["accepted"] / c["actual_drafts"] if c["actual_drafts"] else math.nan
                for c in cycles
            ],
            label="accepted / proposed",
        )
        axes[1, 1].set_ylim(-0.05, 1.05)
        axes[1, 1].set_ylabel("Draft acceptance (no draft = missing)")
        for ax in axes.flat:
            ax.set_xlabel("Completed cycle")
        fig.suptitle(row["prompt_id"] + " — fresh detailed diagnostic request")
        fig.tight_layout()
        fig.savefig(
            directory / "analysis" / (row["prompt_id"] + "-trajectory.png"), dpi=160
        )
        plt.close(fig)
