"""Exploratory linear transition composition; work only, no published-tree claim."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from analysis.frequency_copy import frontier
from analysis.logit_copy import (
    load_collection,
    past_hint,
    replay_canonical,
    guided_draft,
    WORK,
)
from analysis.proposal_replay import suffix_draft, upstream_ngram
from bench.common import ROOT, atomic_json, sha256, source_fingerprint

PROTOCOL = ROOT / "configs/transition-replay-v1.json"
PROTOCOL_SHA256 = "07f2a640e2569049a89792a54214040fcd513a0616216ba7e2a9ced2b7dd0b85"
PRIORITY_PROTOCOL = ROOT / "configs/retrieval-priority-v1.json"
PRIORITY_SHA256 = "1cc45b5095d78e34b4ceb51859604f4d9bcbef629b15c8cc39d5418ece81af25"


def transition_draft(history, pending, cap, *, order):
    if cap not in range(9) or order not in (1, 2):
        raise ValueError("Unsupported linear transition configuration")
    observed = list(history) + [pending]
    # This graph contains only fully observed transitions. Proposed transitions
    # never become evidence, even when walking an already observed cycle.
    edges = {
        tuple(observed[index - order : index]): observed[index]
        for index in range(order, len(observed))
    }
    virtual = observed[-order:]
    proposal = []
    for _ in range(cap):
        token = edges.get(tuple(virtual))
        if token is None:
            break
        proposal.append(token)
        virtual = (virtual + [token])[-order:]
    return proposal


def analyze(collection, out, family="transition"):
    if family not in ("transition", "priority"):
        raise ValueError("Unknown exploratory proposal family")
    protocol_path = PROTOCOL if family == "transition" else PRIORITY_PROTOCOL
    expected_protocol_sha = (
        PROTOCOL_SHA256 if family == "transition" else PRIORITY_SHA256
    )
    targets, receipt = load_collection(collection)
    if sha256(protocol_path) != expected_protocol_sha:
        raise ValueError("Exploratory protocol changed")
    protocol = json.loads(protocol_path.read_text())
    source = source_fingerprint()
    if family == "transition":
        configs = [
            {"id": f"recent-transition{order}", "order": order, "cap": cap}
            for order in protocol["transition_orders"]
            for cap in protocol["transition_caps"]
        ]
        configs += [{"id": "previous-second-logit", "cap": 1}]
    else:
        configs = [
            {"id": "suffix5-priority-" + fallback, "cap": cap}
            for fallback in ("ngram3", "guided3")
            for cap in protocol["caps"]
        ]
        configs += [{"id": "guided3", "cap": cap} for cap in protocol["caps"]]
    configs += [
        {"id": name, "cap": cap}
        for name in ("upstream-ngram3", "suffix5")
        for cap in protocol.get("control_caps", protocol.get("caps"))
    ]
    configs += [{"id": "ordinary", "cap": 0}]
    result, traces = [], []
    for target in targets:
        for config in configs:
            hints = []
            proposals = []

            def propose(history, pending, cap):
                cursor = len(history) + 1 - len(target["prompt_ids"])
                hint_index = None
                if config["id"].startswith("suffix5-priority-"):
                    value = suffix_draft(history, pending, cap, minimum=5)
                    if not value and cap:
                        if config["id"].endswith("ngram3"):
                            value = upstream_ngram(history, pending, cap)
                        else:
                            hint = past_hint(target["rows"], cursor, pending)
                            value = guided_draft(
                                history,
                                pending,
                                cap,
                                hint,
                                minimum=3,
                                cutoff=64,
                                fallback=False,
                            )
                            hint_index = cursor - 1 if hint is not None else None
                elif config["id"] == "guided3":
                    hint = past_hint(target["rows"], cursor, pending)
                    value = guided_draft(
                        history,
                        pending,
                        cap,
                        hint,
                        minimum=3,
                        cutoff=64,
                        fallback=False,
                    )
                    hint_index = cursor - 1 if hint is not None else None
                elif "order" in config:
                    value = transition_draft(
                        history, pending, cap, order=config["order"]
                    )
                elif config["id"] == "previous-second-logit":
                    hint = past_hint(target["rows"], cursor, pending)
                    value = [hint[1]] if hint is not None and cap else []
                    hint_index = cursor - 1 if hint is not None else None
                elif config["id"] == "upstream-ngram3":
                    value = upstream_ngram(history, pending, cap)
                else:
                    value = suffix_draft(history, pending, cap, minimum=5)
                hints.append(hint_index)
                proposals.append(value)
                return value

            if config["id"] == "ordinary":
                n = len(target["token_ids"])
                counts = dict(
                    tokens=n,
                    cycles=n,
                    drafted=0,
                    accepted=0,
                    verified_rows=n,
                    wasted_rows=0,
                )
                cycles = [
                    dict(output_index=i, cap=0, drafted=0, accepted=0, committed=1)
                    for i in range(n)
                ]
                proposals, hints = [[] for _ in cycles], [None for _ in cycles]
            else:
                counts, cycles = replay_canonical(target, config["cap"], propose)
            result.append(
                {
                    "prompt_id": target["prompt_id"],
                    "configuration": config,
                    **counts,
                    "query_size_histogram": dict(
                        sorted(Counter(c["drafted"] + 1 for c in cycles).items())
                    ),
                }
            )
            traces.extend(
                {
                    "prompt_id": target["prompt_id"],
                    "configuration": config,
                    **cycle,
                    "proposals": proposal,
                    "hint_output_index": hint,
                }
                for cycle, proposal, hint in zip(cycles, proposals, hints)
            )
    totals = []
    for config in configs:
        rows = [row for row in result if row["configuration"] == config]
        totals.append(
            {**config, **{name: sum(row[name] for row in rows) for name in WORK}}
        )
    report = {
        "scope": protocol["scope"],
        "source": source,
        "model_id": receipt["model_id"],
        "collection_phase": receipt.get("collection_phase", "fit"),
        "platform": receipt["platform"],
        "collection_complete_sha256": sha256(Path(collection) / "complete.json"),
        "protocol_sha256": sha256(protocol_path),
        "proposal_family": family,
        "totals": totals,
        "per_prompt": result,
        "work_frontier": frontier(
            [
                {
                    **row,
                    "family": (
                        "linear-transition"
                        if row["id"].startswith("recent-transition")
                        else (
                            "suffix-priority"
                            if row["id"].startswith("suffix5-priority-")
                            else "control"
                        )
                    ),
                }
                for row in totals
            ]
        ),
        "computed_cycles": len(traces),
        "performance_claim": None,
    }
    if source_fingerprint() != source:
        raise ValueError("Source changed during replay")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "report.json", report)
    with (out / "cycles.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
            for row in traces:
                zipped.write((json.dumps(row, sort_keys=True) + "\n").encode())
    atomic_json(
        out / "complete.json",
        {
            "sha256": {
                name: sha256(out / name) for name in ("report.json", "cycles.jsonl.gz")
            }
        },
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=("transition", "priority"), default="transition"
    )
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.collection, args.out, args.family)
    print(
        json.dumps(
            {
                "model_id": report["model_id"],
                "phase": report["collection_phase"],
                "computed_cycles": report["computed_cycles"],
                "totals": report["totals"],
            },
            indent=2,
        )
    )
