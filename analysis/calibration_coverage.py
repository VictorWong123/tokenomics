"""Describe the observed states behind a frozen controller prior; no inference."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from bench.common import atomic_json, read_jsonl, sha256
from bench.controller import calibrate


def audit(fit, replay, drafter):
    fit, replay = Path(fit), Path(replay)
    # Re-run the existing identity, exact-state and immutable-artifact audit.
    frozen = json.loads((fit / "run_manifest.json").read_text())["frozen"]
    synthetic = bool(frozen.get("identity", {}).get("synthetic_fixture", False))
    prior = calibrate(fit, replay, drafter, synthetic_fixture=synthetic)
    groups = defaultdict(list)
    for receipt in sorted(replay.glob("*/*/complete.json")):
        directory = receipt.parent
        requests = read_jsonl(directory / "requests.jsonl")
        cycles = read_jsonl(directory / "cycles.jsonl")
        for row in requests:
            cid = row["configuration_id"]
            if not cid.startswith("action") or cid == "actiondynamic":
                continue
            action = int(cid.removeprefix("action"))
            selected = [
                c
                for c in cycles
                if (c["prompt_id"], c["configuration_id"], c["repetition"])
                == (row["prompt_id"], cid, row["repetition"])
            ]
            if len(selected) != 1:
                raise ValueError("Expected one cycle per calibrated continuation")
            groups[action].append((row, selected[0]))
    actions = []
    for action in (0, 1, 2, 4, 6, 8):
        samples = groups[action]
        states = {(r["prompt_id"], r["checkpoint"]) for r, _ in samples}
        with_proposal = {
            (r["prompt_id"], r["checkpoint"])
            for r, c in samples
            if c["actual_drafts"] > 0
        }
        with_acceptance = {
            (r["prompt_id"], r["checkpoint"]) for r, c in samples if c["accepted"] > 0
        }
        actions.append(
            {
                "action": action,
                "timing_samples": len(samples),
                "distinct_states": len(states),
                "states_with_proposals": len(with_proposal),
                "states_with_acceptance": len(with_acceptance),
                "drafted": sum(c["actual_drafts"] for _, c in samples),
                "accepted": sum(c["accepted"] for _, c in samples),
                "checkpoints": sorted({r["checkpoint"] for r, _ in samples}),
                "states": [
                    {"prompt_id": pid, "checkpoint": cp} for pid, cp in sorted(states)
                ],
            }
        )
    return {
        "schema_version": 1,
        "state": "calibration_coverage_audited",
        "prior_id": prior["prior_id"],
        "fit_manifest_sha256": sha256(fit / "run_manifest.json"),
        "replay_manifest_sha256": sha256(replay / "manifest.json"),
        "drafter": drafter,
        "synthetic_fixture": synthetic,
        "actions": actions,
        "interpretation": "Timing repetitions at one checkpoint are repeated observations of one state. They do not supply independent proposal contexts. Missing proposal/acceptance states limit what an unconditional action mean can describe.",
        "speed_claim": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--drafter", choices=("model", "ngram", "suffix"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite an existing coverage report")
    result = audit(args.fit, args.replay, args.drafter)
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
