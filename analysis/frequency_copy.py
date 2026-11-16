"""Exploratory causal frequency-copy work replay; never a speed measurement."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import itertools
import json
from pathlib import Path
import tempfile

from analysis.proposal_replay import (
    CPU_FIT,
    MAC_ARCHIVE,
    analyze_fit,
    extract_mac_fit,
    load_fit,
    replay,
    suffix_draft,
    upstream_ngram,
)
from bench.common import ROOT, atomic_json, sha256, source_fingerprint

PROTOCOL = ROOT / "configs/frequency-copy-fit-v1.json"
PROTOCOL_SHA256 = "77d47a56f6078e91bc290be68b14b6af33cd5bf9b22e539264ab83fc9abefa33"
WORK = ("tokens", "cycles", "drafted", "accepted", "verified_rows", "wasted_rows")


@lru_cache(maxsize=4096)
def occurrences(observed, maximum):
    """Computational cache only: each result searches this exact observed prefix."""
    rows = []
    for length in range(min(maximum, len(observed) - 1), 0, -1):
        pattern = observed[-length:]
        continuations = tuple(
            observed[start + length : start + length + 8]
            for start in range(len(observed) - length - 1, -1, -1)
            if observed[start : start + length] == pattern
        )
        if continuations:
            rows.append((length, continuations))
    return tuple(rows)


def frequency_draft(history, pending, cap, *, minimum, support, confidence, maximum=32):
    numerator, denominator = confidence
    if (
        cap not in range(9)
        or not 1 <= minimum <= maximum <= 32
        or support < 1
        or not 1 <= numerator <= denominator
    ):
        raise ValueError("Invalid frequency-copy settings")
    if cap == 0:
        return []
    observed = tuple(history) + (pending,)
    for length, continuations in occurrences(observed, maximum):
        if length < minimum:
            break
        root_count = len(continuations)
        active = continuations
        proposal = []
        for index in range(cap):
            counts = Counter(seq[index] for seq in active if index < len(seq))
            if not counts:
                break
            # Counter insertion order is the most recent supporting occurrence.
            best = max(counts, key=counts.get)
            count = counts[best]
            if count < support or count * denominator < root_count * numerator:
                break
            proposal.append(best)
            active = tuple(
                seq for seq in active if index < len(seq) and seq[index] == best
            )
        if proposal:
            return proposal
    return []


def settings(protocol):
    for minimum, support, confidence in itertools.product(
        protocol["minimum_match_lengths"],
        protocol["minimum_support_counts"],
        protocol["joint_confidence_fractions"],
    ):
        yield {
            "id": f"freq-m{minimum}-s{support}-p{confidence[0]}of{confidence[1]}",
            "minimum": minimum,
            "support": support,
            "confidence": confidence,
            "maximum": protocol["maximum_match_length"],
        }


def frontier(cells):
    """Unique work points; aliases are not independent experiments."""
    grouped = defaultdict(list)
    for row in cells:
        grouped[(row["cycles"], row["wasted_rows"])].append(
            {key: row[key] for key in ("family", "id", "cap")}
        )
    return [
        {"cycles": cycle, "wasted_rows": waste, "aliases": grouped[(cycle, waste)]}
        for cycle, waste in sorted(grouped)
        if not any(
            c <= cycle and w <= waste and (c < cycle or w < waste) for c, w in grouped
        )
    ]


def analyze(directory, source_id, protocol):
    audited = analyze_fit(directory, source_id)
    _, targets, identity = load_fit(directory)
    occurrences.cache_clear()
    rows, traces = [], []
    for target in targets:
        for configuration in settings(protocol):
            params = {k: v for k, v in configuration.items() if k != "id"}
            for cap in protocol["caps"]:
                proposals = []

                def propose(history, pending, budget):
                    result = frequency_draft(history, pending, budget, **params)
                    proposals.append(result)
                    return result

                counts, cycles = replay(
                    target["prompt_ids"], target["token_ids"], cap, propose
                )
                rows.append(
                    {
                        "prompt_id": target["prompt_id"],
                        "configuration": configuration,
                        "cap": cap,
                        **{key: counts[key] for key in WORK},
                        "verification_rows_histogram": dict(
                            sorted(Counter(c["drafted"] + 1 for c in cycles).items())
                        ),
                    }
                )
                traces.extend(
                    {
                        "source_id": source_id,
                        "family": "frequency",
                        "prompt_id": target["prompt_id"],
                        "id": configuration["id"],
                        "requested_cap": cap,
                        **cycle,
                        "proposal_ids": proposal,
                    }
                    for cycle, proposal in zip(cycles, proposals, strict=True)
                )
    totals = []
    for configuration in settings(protocol):
        for cap in protocol["caps"]:
            selected = [
                r
                for r in rows
                if r["configuration"] == configuration and r["cap"] == cap
            ]
            totals.append(
                {
                    "family": "frequency",
                    "id": configuration["id"],
                    "configuration": configuration,
                    "cap": cap,
                    **{key: sum(r[key] for r in selected) for key in WORK},
                    "verification_rows_histogram": {
                        size: sum(
                            r["verification_rows_histogram"].get(size, 0)
                            for r in selected
                        )
                        for size in range(1, 10)
                        if any(
                            size in r["verification_rows_histogram"] for r in selected
                        )
                    },
                }
            )
            histogram = totals[-1]["verification_rows_histogram"]
            if (
                sum(histogram.values()) != totals[-1]["cycles"]
                or sum(size * count for size, count in histogram.items())
                != totals[-1]["verified_rows"]
            ):
                raise AssertionError("Query-shape accounting differs from replay work")
    # Matched-length raw suffixes distinguish filtering from match-length effects.
    strategies = {"upstream-ngram3": upstream_ngram}
    strategies.update(
        {
            f"suffix{minimum}": lambda h, p, c, m=minimum: suffix_draft(
                h, p, c, minimum=m
            )
            for minimum in protocol["minimum_match_lengths"]
        }
    )
    controls = []
    archived = {(r["strategy"], r["cap"]): r for r in audited["totals"]}
    for name, drafter in strategies.items():
        for cap in protocol["caps"]:
            counts = Counter()
            histogram = Counter()
            for target in targets:
                proposals = []

                def propose(history, pending, budget):
                    proposal = drafter(history, pending, budget)
                    proposals.append(proposal)
                    return proposal

                work, cycles = replay(
                    target["prompt_ids"], target["token_ids"], cap, propose
                )
                counts.update({key: work[key] for key in WORK})
                histogram.update(c["drafted"] + 1 for c in cycles)
                traces.extend(
                    {
                        "source_id": source_id,
                        "family": "control",
                        "prompt_id": target["prompt_id"],
                        "id": name,
                        "requested_cap": cap,
                        **cycle,
                        "proposal_ids": proposal,
                    }
                    for cycle, proposal in zip(cycles, proposals, strict=True)
                )
            if (name, cap) in archived and any(
                counts[key] != archived[(name, cap)][key] for key in WORK
            ):
                raise AssertionError(
                    "Control differs from the independently audited replay"
                )
            controls.append(
                {
                    "family": "control",
                    "id": name,
                    "cap": cap,
                    **counts,
                    "verification_rows_histogram": dict(sorted(histogram.items())),
                }
            )
    tokens = sum(len(t["token_ids"]) for t in targets)
    controls.append(
        {
            "family": "control",
            "id": "ordinary",
            "cap": 0,
            "tokens": tokens,
            "cycles": tokens,
            "drafted": 0,
            "accepted": 0,
            "verified_rows": tokens,
            "wasted_rows": 0,
            "verification_rows_histogram": {1: tokens},
        }
    )
    return {
        "source_id": source_id,
        "identity": identity,
        "baseline_policy_checks": audited["baseline_policy_checks"],
        "baseline_detailed_cycle_checks": audited["baseline_detailed_cycle_checks"],
        "rows": rows,
        "totals": totals,
        "controls": controls,
        "control_scope": "Upstream ngram3 work matches measured archived aggregates; raw suffix/frequency alternatives are counterfactual replay. Matched-length suffix controls isolate frequency filtering.",
        "unique_work_points": len({(r["cycles"], r["wasted_rows"]) for r in totals}),
        "pareto": frontier(totals + controls),
    }, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("The predeclared frequency-copy v1 grid changed")
    protocol = json.loads(PROTOCOL.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        mac, mac_traces = analyze(extract_mac_fit(Path(tmp)), "m3-metal-05b", protocol)
    cpu, cpu_traces = analyze(CPU_FIT, "linux-cpu-3b", protocol)
    traces = args.out / "cycles.jsonl.gz"
    with traces.open("wb") as stream, gzip.GzipFile(
        filename="", mode="wb", fileobj=stream, mtime=0
    ) as compressed:
        for row in mac_traces + cpu_traces:
            compressed.write((json.dumps(row, sort_keys=True) + "\n").encode())
    result = {
        "state": "exploratory_fit_replay_complete",
        "source": source_fingerprint(),
        "protocol": protocol,
        "protocol_sha256": sha256(PROTOCOL),
        "protocol_predeclared_commit": "2cf2be3",
        "mac_archive_sha256": sha256(MAC_ARCHIVE),
        "cycles_sha256": sha256(traces),
        "cycle_records": len(mac_traces) + len(cpu_traces),
        "measured_speedup": None,
        "selection": None,
        "scope": "Three development-fit prompts per operating point,64 tokens each. Current observed request prefix only. No validation or reserved held-out data, model inference or latency estimate.",
        "limitation": "Work-only Pareto points ignore lookup and backend costs. Grid aliases on the same small trajectories are not independent experiments. No selected deployable winner or generalization claim.",
        "sources": [mac, cpu],
    }
    atomic_json(args.out / "frequency-fit.json", result)
    for source in result["sources"]:
        print(source["source_id"], "unique work points", source["unique_work_points"])
        for point in source["pareto"]:
            print(point)


if __name__ == "__main__":
    main()
