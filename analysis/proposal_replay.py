"""Replay causal proposal strategies on archived development-fit greedy trajectories.

Counts are counterfactual work, not measured latency. No model inference runs.
"""

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from analysis.controller import verify
from bench.common import ROOT, atomic_json, read_jsonl, sha256, source_fingerprint

CAPS = (1, 2, 4, 6, 8)
FIT_IDS = {"development-qa-01", "development-code-01", "development-summary-01"}
MAC_ARCHIVE = (
    ROOT / "results/verification/mac-m3-metal-20261111/calibration-and-smoke.tar.gz"
)
CPU_FIT = ROOT / "results/verification/codespace-controller-20261108/attempt-2/fit"


def upstream_ngram(history, pending, cap, n=3):
    """Exact pinned ngram-simple geometry, including its short-match exclusions."""
    if len(history) <= n + 8 + 1 or cap == 0:
        return []
    pattern = history[-(n - 1) :] + [pending] if n > 1 else [pending]
    for start in range(len(history) - n - 1, 0, -1):
        if history[start : start + n] == pattern:
            suffix = history[start + n : start + n + 8]
            return suffix[:cap] if len(suffix) >= n else []
    return []


def suffix_draft(history, pending, cap, minimum=3, maximum=32, consensus=False):
    """Request-local longest suffix; no corpus, future output or cross-request state.

    Search excludes the current occurrence but permits overlap with it. Most
    recent match breaks ties; consensus requires identical next-token evidence
    in at least two earlier occurrences, truncating when that evidence ends.
    """
    if cap == 0:
        return []
    observed = history + [pending]
    for length in range(min(maximum, len(observed) - 1), minimum - 1, -1):
        pattern = observed[-length:]
        matches = [
            start
            for start in range(len(observed) - length - 1, -1, -1)
            if observed[start : start + length] == pattern
        ]
        if not matches:
            continue
        continuations = [
            observed[start + length : start + length + cap] for start in matches
        ]
        if not consensus:
            return continuations[0]
        draft = []
        active = continuations
        for index in range(cap):
            choices = Counter(seq[index] for seq in active if index < len(seq))
            if not choices:
                break
            # Deterministic majority, then most recent occurrence in a tie.
            best = max(
                choices,
                key=lambda token: (
                    choices[token],
                    -next(
                        i
                        for i, seq in enumerate(active)
                        if index < len(seq) and seq[index] == token
                    ),
                ),
            )
            if choices[best] < 2:
                break
            draft.append(best)
            active = [seq for seq in active if index < len(seq) and seq[index] == best]
        if draft:
            return draft
    return []


STRATEGIES = {
    "upstream-ngram3": upstream_ngram,
    "upstream-ngram2": lambda h, p, c: upstream_ngram(h, p, c, n=2),
    "suffix3": suffix_draft,
    "suffix2": lambda h, p, c: suffix_draft(h, p, c, minimum=2),
    "suffix4": lambda h, p, c: suffix_draft(h, p, c, minimum=4),
    "suffix5": lambda h, p, c: suffix_draft(h, p, c, minimum=5),
    "suffix6": lambda h, p, c: suffix_draft(h, p, c, minimum=6),
    "suffix8": lambda h, p, c: suffix_draft(h, p, c, minimum=8),
    "suffix12": lambda h, p, c: suffix_draft(h, p, c, minimum=12),
    "consensus2": lambda h, p, c: suffix_draft(h, p, c, minimum=2, consensus=True),
}


def replay(prompt, output, cap, propose):
    if not prompt or not output or cap not in CAPS:
        raise ValueError("Nonempty canonical trajectory and supported cap required")
    history, pending = list(prompt[:-1]), prompt[-1]
    index = 0
    cycles = []
    while index < len(output):
        # Reserve one correction/bonus token, matching draft_budget exactly.
        effective = min(cap, len(output) - index - 1)
        proposals = propose(history, pending, effective)
        if len(proposals) > effective:
            raise ValueError("Proposal exceeded effective budget")
        accepted = 0
        for token, expected in zip(proposals, output[index:]):
            if token != expected:
                break
            accepted += 1
        committed = accepted + 1
        cycles.append(
            {
                "output_index": index,
                "cap": effective,
                "drafted": len(proposals),
                "accepted": accepted,
                "committed": committed,
            }
        )
        # Only committed canonical tokens become future proposal input.
        for token in output[index : index + committed]:
            history.append(pending)
            pending = token
        index += committed
    counts = {
        "tokens": len(output),
        "cycles": len(cycles),
        "drafted": sum(c["drafted"] for c in cycles),
        "accepted": sum(c["accepted"] for c in cycles),
    }
    counts["verified_rows"] = counts["cycles"] + counts["drafted"]
    counts["wasted_rows"] = counts["verified_rows"] - counts["tokens"]
    # Stylized cost: C*H + V*R versus N*(H+R), with zero drafting cost.
    # H is a batch-fixed cost, R a per-row cost. This is an optimistic bound.
    saved = counts["tokens"] - counts["cycles"]
    counts["required_fixed_cost_fraction"] = (
        counts["wasted_rows"] / (counts["wasted_rows"] + saved) if saved else None
    )
    return counts, cycles


def load_fit(directory):
    verify(directory)
    rows = read_jsonl(Path(directory) / "requests.jsonl")
    manifest = json.loads((Path(directory) / "run_manifest.json").read_text())
    frozen = manifest["frozen"]
    corpus = {
        p["id"]: p
        for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")
        if p["id"] in FIT_IDS
    }
    if any(p != corpus.get(p["id"]) for p in frozen["prompts"]):
        raise ValueError("Fit prompt text or metadata differs from the frozen corpus")
    if {p["id"] for p in frozen["prompts"]} != FIT_IDS or any(
        r["prompt_id"] not in FIT_IDS
        or r["output_cap"] != 64
        or r["repetition"] != 0
        or r["stop"] != "budget"
        or len(r["token_ids"]) != 64
        for r in rows
    ):
        raise ValueError(
            "Only the predeclared three fit prompts, cap64, repetition0, budget stops are eligible"
        )
    targets = [r for r in rows if r["configuration_id"] == "target"]
    if len(targets) != 3:
        raise ValueError("Exactly three canonical target records required")
    return (
        rows,
        targets,
        {
            "run_manifest_sha256": sha256(Path(directory) / "run_manifest.json"),
            "requests_sha256": sha256(Path(directory) / "requests.jsonl"),
            "target_sha256": frozen["target_sha256"],
            "gpu_layers": frozen["gpu_layers"],
        },
    )


def analyze_fit(directory, source_id):
    rows, targets, identity = load_fit(directory)
    measured = {(r["prompt_id"], r["configuration_id"]): r for r in rows}
    results = []
    baseline_checks = 0
    cycle_checks = 0
    for target in targets:
        for name, propose in STRATEGIES.items():
            for cap in CAPS:
                counts, cycles = replay(
                    target["prompt_ids"], target["token_ids"], cap, propose
                )
                if name == "upstream-ngram3":
                    observed = measured[(target["prompt_id"], f"fixed{cap}")]
                    if (
                        observed["prompt_ids"] != target["prompt_ids"]
                        or observed["token_ids"] != target["token_ids"]
                    ):
                        raise ValueError(
                            "Measured baseline differs from ordinary output"
                        )
                    for key in ("cycles", "drafted", "accepted", "verified_rows"):
                        if counts[key] != observed["execution"][key]:
                            raise ValueError(
                                f"Replay disagrees with measured upstream {key}"
                            )
                    archived = read_jsonl(
                        Path(directory)
                        / "blocks"
                        / observed["block_id"]
                        / "attempt-0"
                        / f"fixed{cap}.cycles.jsonl"
                    )
                    if archived and (
                        len(archived) != len(cycles)
                        or any(
                            (a["actual_drafts"], a["accepted"], a["committed"])
                            != (b["drafted"], b["accepted"], b["committed"])
                            for a, b in zip(archived, cycles)
                        )
                    ):
                        raise ValueError("Replay cycle-by-cycle baseline check failed")
                    cycle_checks += bool(archived)
                    baseline_checks += 1
                results.append(
                    {
                        "source_id": source_id,
                        "prompt_id": target["prompt_id"],
                        "strategy": name,
                        "cap": cap,
                        **counts,
                    }
                )
    totals = []
    for name in STRATEGIES:
        for cap in CAPS:
            selected = [
                row for row in results if row["strategy"] == name and row["cap"] == cap
            ]
            counts = {
                key: sum(row[key] for row in selected)
                for key in (
                    "tokens",
                    "cycles",
                    "drafted",
                    "accepted",
                    "verified_rows",
                    "wasted_rows",
                )
            }
            saved = counts["tokens"] - counts["cycles"]
            counts["required_fixed_cost_fraction"] = (
                counts["wasted_rows"] / (counts["wasted_rows"] + saved)
                if saved
                else None
            )
            totals.append({"strategy": name, "cap": cap, **counts})
    return {
        "source_id": source_id,
        "identity": identity,
        "baseline_policy_checks": baseline_checks,
        "baseline_detailed_cycle_checks": cycle_checks,
        "rows": results,
        "totals": totals,
    }


def extract_mac_fit(destination):
    # Read only regular fit members. Do not unpack validation or unrelated logs.
    with tarfile.open(MAC_ARCHIVE) as archive:
        seen = set()
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if not path.parts or path.parts[0] != "fit" or member.isdir():
                continue
            if (
                path.is_absolute()
                or ".." in path.parts
                or not member.isfile()
                or member.name in seen
            ):
                raise ValueError("Unsafe or duplicate fit archive member")
            seen.add(member.name)
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    return destination / "fit"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as tmp:
        mac = analyze_fit(extract_mac_fit(Path(tmp)), "m3-metal-05b")
    cpu = analyze_fit(CPU_FIT, "linux-cpu-3b")
    result = {
        "schema_version": 1,
        "source": source_fingerprint(),
        "mac_archive_sha256": sha256(MAC_ARCHIVE),
        "state": "offline_replay_complete",
        "measured_speedup": None,
        "scope": "Three predeclared development-fit prompts per operating point, 64 tokens each. Alternative proposals see only the current prompt/committed history. No held-out data or inference.",
        "cost_model": "Optimistic H + rows*R per cycle with zero drafting cost; required_fixed_cost_fraction = wasted_rows / (wasted_rows + saved_cycles). Null means no cycle saving, so no strict gain under this model.",
        "sources": [mac, cpu],
    }
    atomic_json(args.out / "replay.json", result)
    for source in result["sources"]:
        print(source["source_id"], "baseline checks", source["baseline_policy_checks"])
        for row in source["totals"]:
            if row["cap"] in (1, 4, 8):
                print(row)


if __name__ == "__main__":
    main()
