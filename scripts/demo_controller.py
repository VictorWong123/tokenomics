#!/usr/bin/env python3
"""Read a verified controller trajectory without downloading models or inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.controller import verify
from bench.common import ROOT, read_jsonl, sha256
from bench.controller import calibrate


def recorded_directory():
    root = ROOT / "results/verification/codespace-controller-20261108"
    receipt = json.loads((root / "verification.json").read_text())
    if (
        not receipt["independent_output_checks"]
        or not receipt["shutdown_confirmed"]
        or receipt["state"] != "controller_study_complete"
    ):
        raise ValueError("A completed independently checked controller run is required")
    directory = root / f"attempt-{receipt['attempt']}"
    manifest = json.loads((directory / "execution-manifest.json").read_text())
    if any(
        str(manifest[k]) != str(receipt[r])
        for k, r in (
            ("run_id", "run_id"),
            ("attempt", "attempt"),
            ("sha", "source_sha"),
        )
    ):
        raise ValueError(
            "Controller archive ownership differs from verification receipt"
        )
    for name, entry in manifest["files"].items():
        path = directory / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError("Retained controller evidence changed")
    return directory


def render(directory, *, prompt_id=None, max_cycles=24):
    rows, manifest = verify(directory / "logging")
    fixture = manifest["frozen"]["identity"].get("synthetic_fixture", False)
    prior = json.loads((directory / "prior.json").read_text())
    rebuilt = calibrate(
        directory / "fit",
        directory / "replay",
        prior["provenance"]["operating_point"]["drafter"],
        synthetic_fixture=fixture,
    )
    if prior != rebuilt:
        raise ValueError("Controller prior differs from retained calibration")
    selected = [
        r
        for r in rows
        if r["configuration_id"] == "cost-aware-detailed"
        and r["repetition"] == 0
        and (prompt_id is None or r["prompt_id"] == prompt_id)
    ]
    if not selected:
        raise ValueError("No detailed controller trace for that prompt")
    lines = [
        "Experimental C++ cost-aware controller",
        (
            "Synthetic transformer fixtures; not trained-model performance."
            if fixture
            else "Recorded trained-model development trace; no inference executed."
        ),
        "Fresh detailed requests selected by the declared median/worst rule.",
        "prior: " + prior["prior_id"],
        "0 = target-only; probe does not replace the normal incumbent.",
    ]
    for row in selected:
        cycles = [
            c
            for c in read_jsonl(directory / "logging" / row["source_cycles"])
            if not c["warmup"]
        ]
        lines += [
            "",
            f"{row['prompt_id']}: {row['generated_tokens']} exact tokens, stop={row['stop']}",
            "cycle action actual accept commit stale sync  predicted_ms observed_ms reason",
        ]
        indexes = list(range(len(cycles)))
        if max_cycles and len(indexes) > max_cycles:
            # Include the beginning, end and observed action/probe transitions.
            important = [
                i
                for i, c in enumerate(cycles)
                if i == 0
                or i == len(cycles) - 1
                or c["controller"]["probe"]
                or c["action"] != cycles[i - 1]["action"]
            ]
            indexes = sorted(set(important[: max_cycles - 1] + [len(cycles) - 1]))
        for i in indexes:
            c = cycles[i]
            t = c["controller"]
            reason = t["reason"] + ("*" if t["probe"] else "")
            lines.append(
                f"{c['cycle']:5d} {c['action']:6d} {c['actual_drafts']:6d} {c['accepted']:6d} {c['committed']:6d} {c['stale_before']:5d} {c['synced']:4d} {t['predicted_ms']:13.3f} {c['observation_ms']:11.3f} {reason}"
            )
        if len(indexes) != len(cycles):
            lines.append(
                f"Showing {len(indexes)}/{len(cycles)} cycles; --max-cycles 0 displays every cycle."
            )
        lines.append(
            f"probes={row['controller']['probes']}, fallback_tokens={row['controller']['fallback_tokens']}, forced_tails={row['controller']['forced_tail_cycles']}; generation={row['generation_ms']:.2f} ms (all charged)"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Complete local study; default is retained trained evidence",
    )
    parser.add_argument("--prompt-id")
    parser.add_argument("--max-cycles", type=int, default=24)
    args = parser.parse_args()
    if args.max_cycles < 0 or args.max_cycles == 1:
        parser.error("max-cycles must be 0 (all) or at least 2")
    directory = args.run_dir if args.run_dir else recorded_directory()
    print(
        render(directory, prompt_id=args.prompt_id, max_cycles=args.max_cycles), end=""
    )
