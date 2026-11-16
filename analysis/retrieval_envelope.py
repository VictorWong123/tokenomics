"""Optimistic request-local copying limit, using future-aware oracle selection.

This is a diagnostic bound, never a deployable drafter or a speed prediction.
Candidate contents come only from observed tokens; the oracle chooses their
length and occurrence using the canonical future. No reserved held-out data.
"""

import argparse
from pathlib import Path
import tempfile

from analysis.proposal_replay import (
    CAPS,
    CPU_FIT,
    MAC_ARCHIVE,
    analyze_fit,
    extract_mac_fit,
    load_fit,
)
from bench.common import atomic_json, sha256, source_fingerprint


def copy_options(observed, cap, minimum, maximum=32):
    """All allowed contiguous-copy proposals, including their shorter prefixes."""
    if minimum < 1 or maximum < minimum or cap < 0 or not observed:
        raise ValueError("Invalid copy envelope settings")
    options = {()}
    for length in range(minimum, min(maximum, len(observed) - 1) + 1):
        suffix = observed[-length:]
        for start in range(len(observed) - length):
            if observed[start : start + length] != suffix:
                continue
            continuation = observed[start + length : start + length + cap]
            for count in range(1, len(continuation) + 1):
                options.add(tuple(continuation[:count]))
    return options


def envelope(prompt, output, cap, minimum):
    if not prompt or not output or cap not in CAPS:
        raise ValueError("Nonempty canonical trajectory and supported cap required")
    best = []
    for index in range(len(output)):
        options = copy_options(
            prompt + output[:index], min(cap, len(output) - index - 1), minimum
        )
        # Selection reads the future intentionally. Proposal contents do not.
        best.append(
            max(
                len(option)
                for option in options
                if list(option) == output[index : index + len(option)]
            )
        )
    cost = [0] * (len(output) + 1)
    chosen = [0] * len(output)
    for index in range(len(output) - 1, -1, -1):
        cost[index], negative = min(
            (1 + cost[index + accepted + 1], -accepted)
            for accepted in range(best[index] + 1)
        )
        chosen[index] = -negative
    index = 0
    trace = []
    while index < len(output):
        accepted = chosen[index]
        trace.append({"output_index": index, "copied": accepted})
        index += accepted + 1
    if len(trace) != cost[0] or sum(row["copied"] + 1 for row in trace) != len(output):
        raise AssertionError("Oracle path failed to consume the canonical trajectory")
    return {
        "tokens": len(output),
        "minimum_cycles": cost[0],
        "maximum_saved_cycles": len(output) - cost[0],
        "verified_rows": len(output),
        "wasted_rows": 0,
        "trace": trace,
    }


def analyze(directory, source_id):
    audited = analyze_fit(directory, source_id)
    _, targets, identity = load_fit(directory)
    rows = []
    for target in targets:
        for minimum in (1, 3, 5):
            for cap in CAPS:
                result = envelope(
                    target["prompt_ids"], target["token_ids"], cap, minimum
                )
                if minimum == 5:
                    baseline = next(
                        row
                        for row in audited["rows"]
                        if row["prompt_id"] == target["prompt_id"]
                        and row["strategy"] == "suffix5"
                        and row["cap"] == cap
                    )
                    if result["minimum_cycles"] > baseline["cycles"]:
                        raise AssertionError(
                            "Oracle is worse than an eligible strategy"
                        )
                    result["current_suffix_cycles"] = baseline["cycles"]
                    result["current_suffix_wasted_rows"] = baseline["wasted_rows"]
                rows.append(
                    {
                        "prompt_id": target["prompt_id"],
                        "minimum_suffix": minimum,
                        "cap": cap,
                        **result,
                    }
                )
    totals = []
    for minimum in (1, 3, 5):
        for cap in CAPS:
            selected = [
                r for r in rows if r["minimum_suffix"] == minimum and r["cap"] == cap
            ]
            keys = ["tokens", "minimum_cycles", "maximum_saved_cycles", "verified_rows"]
            if minimum == 5:
                keys += ["current_suffix_cycles", "current_suffix_wasted_rows"]
            totals.append(
                {
                    "minimum_suffix": minimum,
                    "cap": cap,
                    **{key: sum(r[key] for r in selected) for key in keys},
                }
            )
    return {
        "source_id": source_id,
        "identity": identity,
        "baseline_policy_checks": audited["baseline_policy_checks"],
        "rows": rows,
        "totals": totals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as tmp:
        mac = analyze(extract_mac_fit(Path(tmp)), "m3-metal-05b")
    cpu = analyze(CPU_FIT, "linux-cpu-3b")
    result = {
        "state": "offline_oracle_envelope_complete",
        "source": source_fingerprint(),
        "mac_archive_sha256": sha256(MAC_ARCHIVE),
        "measured_speedup": None,
        "future_aware_selection": True,
        "scope": "Three predeclared fit prompts per platform, cap64. No held-out data or inference.",
        "limitation": "The oracle knows the canonical future, incurs no lookup cost and rejects nothing. This is an optimistic copying bound, not a causal strategy or a runtime forecast. Cross-request retrieval and learned drafts lie outside its scope.",
        "sources": [mac, cpu],
    }
    atomic_json(args.out / "envelope.json", result)
    for source in result["sources"]:
        print(source["source_id"])
        for row in source["totals"]:
            if row["cap"] in (1, 8):
                print(row)


if __name__ == "__main__":
    main()
