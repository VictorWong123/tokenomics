"""Causal, canonical-logit copy replay. No latency or published-tree reproduction."""

import argparse
import base64
from collections import Counter
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import re

from analysis.frequency_copy import frontier
from analysis.proposal_replay import replay, suffix_draft, upstream_ngram
from bench.common import ROOT, atomic_json, read_jsonl, sha256, source_fingerprint
from bench.model_identity import model_identity
from scripts.collect_logit_fit import PROTOCOL, PROTOCOL_SHA256

WORK = ("tokens", "cycles", "drafted", "accepted", "verified_rows", "wasted_rows")


def copy_candidate(history, pending, cap, *, minimum, maximum=32, hints=None):
    """Search known source boundaries; hints rank only the proposed first token."""
    if cap not in range(9) or not 1 <= minimum <= maximum <= 32:
        raise ValueError("Invalid copy settings")
    if cap == 0:
        return []
    observed = tuple(history) + (pending,)
    best = None
    for boundary in range(len(observed) - 1, 0, -1):
        token = observed[boundary]
        if hints is not None and token not in hints:
            continue
        length = 0
        while (
            length < min(maximum, boundary)
            and observed[boundary - length - 1] == observed[-length - 1]
        ):
            length += 1
        if length < minimum:
            continue
        score = (length, -hints[token] if hints is not None else 0, boundary)
        if best is None or score > best[0]:
            best = score, boundary
    return list(observed[best[1] : best[1] + cap]) if best else []


def guided_draft(history, pending, cap, prior_top_ids, *, minimum, cutoff, fallback):
    if not 1 <= cutoff <= 64:
        raise ValueError("Invalid hint cutoff")
    if prior_top_ids is not None:
        if not prior_top_ids or prior_top_ids[0] != pending:
            raise ValueError("Prior distribution does not predict the pending token")
        hints = {
            token: rank
            for rank, token in enumerate(prior_top_ids[:cutoff])
            if token != pending
        }
        proposed = copy_candidate(history, pending, cap, minimum=minimum, hints=hints)
        if proposed:
            return proposed
    return copy_candidate(history, pending, cap, minimum=minimum) if fallback else []


def past_hint(rows, cursor, pending):
    if cursor < 0:
        raise ValueError("Negative canonical cursor")
    if cursor == 0:
        return None  # Never read rows[-1] or a first-cycle future distribution.
    row = rows[cursor - 1]
    if row["output_index"] != cursor - 1 or row["top_ids"][0] != pending:
        raise ValueError("Misaligned canonical hint")
    return row["top_ids"]


def import_collection(log, out, run_url, source_commit, *, allow_synthetic=False):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Expected a full workflow source commit")
    if not re.fullmatch(
        r"https://github.com/VictorWong123/tokenomics/actions/runs/[0-9]+", run_url
    ):
        raise ValueError("Expected this repository's public workflow URL")
    lines = Path(log).read_text().splitlines()
    receipts = [
        json.loads(line.split("LOGIT_FIT_RECEIPT=", 1)[1])
        for line in lines
        if "LOGIT_FIT_RECEIPT=" in line
    ]
    if len(receipts) != 1:
        raise ValueError("Expected exactly one completed collection receipt")
    receipt = receipts[0]
    identity = model_identity(receipt["model_id"])
    if (
        receipt.get("state") != "passed"
        or receipt.get("protocol_sha256") != PROTOCOL_SHA256
        or receipt["source"]["revision"] != source_commit
        or (receipt.get("synthetic_fixture") and not allow_synthetic)
        or (
            not receipt.get("synthetic_fixture")
            and receipt["model_sha256"] != identity["sha256"]
        )
    ):
        raise ValueError("Wrong collection identity or synthetic/learned scope")
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Fit protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    records = {}
    for line in lines:
        if "LOGIT_FIT_DATA=" not in line:
            continue
        item = json.loads(line.split("LOGIT_FIT_DATA=", 1)[1])
        size = item["bytes"]
        if not 0 < size <= 128 * 1024 or item["prompt_id"] in records:
            raise ValueError("Oversized or duplicated rank payload")
        packed = base64.b64decode(item["gzip_base64"], validate=True)
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
            raw = stream.read(size + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Rank payload hash/size mismatch")
        row = json.loads(raw)
        if row["prompt_id"] != item["prompt_id"]:
            raise ValueError("Payload prompt ID differs")
        records[item["prompt_id"]] = raw
    if set(records) != set(protocol["prompt_ids"]):
        raise ValueError("Missing or unexpected development-fit prompt")
    data = b"".join(records[pid] + b"\n" for pid in protocol["prompt_ids"])
    if hashlib.sha256(data).hexdigest() != receipt["payload_sha256"]["data.jsonl"]:
        raise ValueError("Combined rank dataset hash differs from sealed collection")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "data.jsonl").write_bytes(data)
    atomic_json(out / "receipt.json", receipt)
    atomic_json(
        out / "provenance.json",
        {
            "public_run_url": run_url,
            "source_commit": source_commit,
            "scope": "Selected numeric rank data and workflow-reported sealed hashes recovered from public logs",
            "limitation": "Original collector stdout/jobs/manifest files are not independently re-downloaded; their sealed hashes are retained in receipt.json",
        },
    )
    atomic_json(
        out / "complete.json",
        {
            "sha256": {
                name: sha256(out / name)
                for name in ("data.jsonl", "receipt.json", "provenance.json")
            }
        },
    )
    return receipt


def load_collection(directory):
    directory = Path(directory)
    seal = json.loads((directory / "complete.json").read_text())["sha256"]
    if set(seal) != {"data.jsonl", "receipt.json", "provenance.json"} or any(
        sha256(directory / name) != digest for name, digest in seal.items()
    ):
        raise ValueError("Imported collection changed")
    receipt = json.loads((directory / "receipt.json").read_text())
    if (
        sha256(directory / "data.jsonl") != receipt["payload_sha256"]["data.jsonl"]
        or receipt["protocol_sha256"] != PROTOCOL_SHA256
    ):
        raise ValueError("Wrong sealed data/protocol")
    identity = model_identity(receipt["model_id"])
    provenance = json.loads((directory / "provenance.json").read_text())
    if (
        receipt.get("state") != "passed"
        or receipt["source"]["revision"] != provenance["source_commit"]
        or (
            not receipt.get("synthetic_fixture")
            and receipt["model_sha256"] != identity["sha256"]
        )
    ):
        raise ValueError("Imported collection identity differs")
    rows = read_jsonl(directory / "data.jsonl")
    protocol = json.loads(PROTOCOL.read_text())
    if [row["prompt_id"] for row in rows] != protocol["prompt_ids"]:
        raise ValueError("Wrong development-fit prompts")
    for row in rows:
        if (
            not row["observer_exact"]
            or not row["prompt_ids"]
            or not 1 <= len(row["token_ids"]) <= 64
            or len(row["rows"]) != len(row["token_ids"])
        ):
            raise ValueError("Incomplete canonical trajectory")
        for i, (token, rank) in enumerate(
            zip(row["token_ids"], row["rows"], strict=True)
        ):
            if (
                rank["output_index"] != i
                or len(rank["top_ids"]) != 64
                or len(set(rank["top_ids"])) != 64
                or rank["top_ids"][0] != token
            ):
                raise ValueError("Invalid canonical ranking")
    return rows, receipt


def analyze(directory, out):
    targets, receipt = load_collection(directory)
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Fit protocol changed")
    source = source_fingerprint()
    configurations = [
        {
            "family": "guided",
            "id": f"hint-m{minimum}-k{cutoff}-f{int(fallback)}",
            "minimum": minimum,
            "cutoff": cutoff,
            "fallback": fallback,
            "cap": cap,
        }
        for minimum, cutoff, fallback, cap in itertools.product(
            protocol["minimum_match_lengths"],
            protocol["hint_rank_cutoffs"],
            protocol["fallback_to_plain_copy"],
            protocol["caps"],
        )
    ]
    configurations += [
        {"family": "control", "id": f"suffix{minimum}", "minimum": minimum, "cap": cap}
        for minimum, cap in itertools.product(
            protocol["minimum_match_lengths"], protocol["caps"]
        )
    ]
    configurations += [
        {"family": "control", "id": "upstream-ngram3", "cap": cap}
        for cap in protocol["caps"]
    ]
    results, traces = [], []
    for config in configurations:
        for target in targets:
            proposals = []
            used_hints = []

            def propose(history, pending, cap):
                cursor = len(history) + 1 - len(target["prompt_ids"])
                hint = (
                    past_hint(target["rows"], cursor, pending)
                    if config["family"] == "guided"
                    else None
                )
                if config["family"] == "guided":
                    value = guided_draft(
                        history,
                        pending,
                        cap,
                        hint,
                        **{
                            key: config[key]
                            for key in ("minimum", "cutoff", "fallback")
                        },
                    )
                elif config["id"] == "upstream-ngram3":
                    value = upstream_ngram(history, pending, cap)
                else:
                    value = suffix_draft(
                        history, pending, cap, minimum=config["minimum"]
                    )
                proposals.append(value)
                used_hints.append(cursor - 1 if hint is not None else None)
                return value

            counts, cycles = replay(
                target["prompt_ids"], target["token_ids"], config["cap"], propose
            )
            histogram = dict(sorted(Counter(c["drafted"] + 1 for c in cycles).items()))
            if (
                sum(histogram.values()) != counts["cycles"]
                or sum(size * count for size, count in histogram.items())
                != counts["verified_rows"]
            ):
                raise AssertionError("Query-shape histogram differs from work")
            results.append(
                {
                    "prompt_id": target["prompt_id"],
                    "configuration": config,
                    **{key: counts[key] for key in WORK},
                    "query_rows_histogram": histogram,
                }
            )
            for cycle, proposal, hint_row in zip(
                cycles, proposals, used_hints, strict=True
            ):
                if hint_row is not None and hint_row >= cycle["output_index"]:
                    raise AssertionError("Future distribution used for drafting")
                traces.append(
                    {
                        "prompt_id": target["prompt_id"],
                        **config,
                        "requested_cap": config["cap"],
                        **cycle,
                        "proposal_ids": proposal,
                        "hint_row_used": hint_row,
                    }
                )
    totals = []
    for config in configurations:
        selected = [r for r in results if r["configuration"] == config]
        histogram = Counter()
        for row in selected:
            histogram.update(row["query_rows_histogram"])
        totals.append(
            {
                **config,
                **{key: sum(r[key] for r in selected) for key in WORK},
                "query_rows_histogram": dict(sorted(histogram.items())),
            }
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "source": source,
        "model_id": receipt["model_id"],
        "collection_source": receipt["source"],
        "dataset_sha256": sha256(Path(directory) / "data.jsonl"),
        "protocol_sha256": PROTOCOL_SHA256,
        "synthetic_fixture": receipt["synthetic_fixture"],
        "scope": "Exploratory canonical-logit chain work replay, not published tree algorithms, latency, validation or held-out evidence",
        "per_prompt": results,
        "totals": totals,
        "work_frontier": frontier(totals),
        "performance_claim": None,
        "runtime_selected": False,
    }
    atomic_json(out / "fit.json", report)
    with (out / "cycles.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for row in traces:
                stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
    if source_fingerprint() != source:
        raise ValueError("Source changed during work replay")
    atomic_json(
        out / "complete.json",
        {
            "sha256": {
                name: sha256(out / name) for name in ("fit.json", "cycles.jsonl.gz")
            }
        },
    )
    print(
        json.dumps(
            {
                "model_id": receipt["model_id"],
                "cells": len(totals),
                "guided_cells": sum(c["family"] == "guided" for c in totals),
                "cycles": len(traces),
                "work_frontier": report["work_frontier"],
                "performance_claim": None,
            },
            sort_keys=True,
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--log", type=Path, required=True)
    imp.add_argument("--run-url", required=True)
    imp.add_argument("--source-commit", required=True)
    imp.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("replay")
    run.add_argument("--collection", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "import":
        import_collection(args.log, args.out, args.run_url, args.source_commit)
    else:
        analyze(args.collection, args.out)


if __name__ == "__main__":
    main()
