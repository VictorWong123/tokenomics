"""Bounded real-model CPU correctness smoke, not a performance experiment."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import (
    ROOT,
    atomic_json,
    hardware,
    read_jsonl,
    sha256,
    source_fingerprint,
)
from scripts.check_metal_profile import GREEDY_MARKER, MODEL_SHA256
from scripts.check_models import check_cycles, compare, execute

WORK = ("cycles", "drafted", "accepted", "verified_rows", "synced_tokens")


def variants():
    yield {"configuration_id": "target", "policy": "target"}
    for drafter in ("ngram", "suffix"):
        for cap in (1, 4, 8):
            yield {
                "configuration_id": f"{drafter}{cap}",
                "policy": "fixed",
                "k": cap,
                "drafter": drafter,
            }
    yield {
        "configuration_id": "script-ngram",
        "policy": "script",
        "k": 8,
        "drafter": "ngram",
        "script": [0, 4, 1, 8, 2, 0, 6],
    }


def smoke(args, *, expected_sha=MODEL_SHA256):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "purpose": "Trained CPU exact-output and work-count smoke; no M3 or speed claim",
        "synthetic_fixture": expected_sha != MODEL_SHA256,
        "source": source_fingerprint(),
        "hardware": hardware(),
        "model_sha256": sha256(args.model),
        "binary_sha256": {
            name: sha256(binary)
            for name, binary in (
                ("reference", args.reference),
                ("greedy", args.candidate),
            )
        },
        "scope": "Four development prompts, tiny output budgets and two KV boundaries; no reserved held-out data",
    }
    atomic_json(out / "manifest.json", report)
    if report["model_sha256"] != expected_sha:
        raise ValueError("Expected the pinned Qwen0.5B model")
    settings = SimpleNamespace(
        binary=args.reference,
        target=args.model,
        out=out,
        context=1024,
        batch=512,
        threads=2,
        gpu_layers=0,
        timeout_seconds=900,
    )
    corpus = {p["id"]: p for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")}
    cases = [
        {"prompt_id": pid, "prompt": corpus[pid]["prompt"], "max_tokens": cap}
        for pid, cap in (
            ("development-code-03", 128),
            ("development-qa-01", 64),
            ("development-code-01", 64),
            ("development-summary-01", 64),
        )
    ]
    cases += [
        {"prompt_id": f"budget-{cap}", "prompt": "a", "max_tokens": cap}
        for cap in (0, 1, 2, 7)
    ]
    checked = 0
    stage = "reference"
    try:
        references = execute(
            settings,
            "reference",
            [
                {
                    **case,
                    "policy": "target",
                    "configuration_id": "target",
                    "repetition": 0,
                }
                for case in cases
            ],
            False,
        )
        seed = references[-1]["prompt_ids"]
        boundaries = [
            {
                "prompt_id": f"prefix-{length}",
                "prefix_tokens": (seed * (length // len(seed) + 1))[:length],
                "max_tokens": 9,
            }
            for length in (255, 511)
        ]
        references += execute(
            settings,
            "boundary-reference",
            [
                {
                    **case,
                    "policy": "target",
                    "configuration_id": "target",
                    "repetition": 0,
                }
                for case in boundaries
            ],
            False,
        )
        expected = {row["prompt_id"]: row for row in references}
        cases += boundaries
        jobs = [
            {**case, **variant, "repetition": 0}
            for case in cases
            for variant in variants()
        ]
        outputs = {}
        with (out / "correctness.jsonl").open("w") as stream:
            for stage, binary, candidate in (
                ("guarded", args.reference, False),
                ("greedy", args.candidate, True),
            ):
                settings.binary = binary

                def check(row):
                    nonlocal checked
                    marker = GREEDY_MARKER in row["system_info"]
                    if (
                        marker != candidate
                        or "TOKENOMICS_CPU_BATCH_INVARIANT=ON" not in row["system_info"]
                    ):
                        raise ValueError("Wrong CPU/sampler profile executed")
                    mismatch = compare(expected[row["prompt_id"]], row)
                    audit = {
                        "stage": stage,
                        "prompt_id": row["prompt_id"],
                        "configuration_id": row["configuration_id"],
                        "mismatch": mismatch,
                    }
                    stream.write(json.dumps(audit) + "\n")
                    stream.flush()
                    checked += 1
                    if mismatch is not None:
                        raise ValueError(
                            "Trained CPU output differs; first divergence retained"
                        )

                outputs[stage] = execute(settings, stage, jobs, False, check)
                check_cycles(outputs[stage], read_jsonl(out / f"{stage}.cycles.jsonl"))
        totals = {}
        for stage, rows in outputs.items():
            totals[stage] = {}
            for row in rows:
                total = totals[stage].setdefault(
                    row["configuration_id"], {key: 0 for key in WORK}
                )
                for key in WORK:
                    total[key] += row["execution"][key]
        for ordinary, candidate in zip(outputs["guarded"], outputs["greedy"]):
            if any(
                ordinary["execution"][key] != candidate["execution"][key]
                for key in WORK
            ):
                raise ValueError(
                    "Sampler profiles did different logical inference work"
                )
        for drafter in ("ngram", "suffix"):
            if not any(
                row["execution"]["drafted"] > 0 and row["execution"]["accepted"] > 0
                for row in outputs["greedy"]
                if row["settings"]["drafter"] == drafter
            ):
                raise ValueError(f"No accepted {drafter} proposals in this smoke")
        if source_fingerprint() != report["source"] or any(
            sha256(binary) != report["binary_sha256"][name]
            for name, binary in (
                ("reference", args.reference),
                ("greedy", args.candidate),
            )
        ):
            raise ValueError("Source or executable changed during smoke")
        status = {
            "state": "passed",
            "checked_comparisons": checked,
            "same_work_between_sampler_profiles": True,
            "totals": totals,
            "performance_claim": None,
            "synthetic_fixture": report["synthetic_fixture"],
        }
        atomic_json(out / "status.json", status)
        names = ["manifest.json", "status.json", "correctness.jsonl"] + [
            f"{name}.{suffix}"
            for name in ("reference", "boundary-reference", "guarded", "greedy")
            for suffix in ("jobs.jsonl", "jsonl", "cycles.jsonl", "stderr.log")
        ]
        atomic_json(
            out / "complete.json",
            {"sha256": {name: sha256(out / name) for name in names}},
        )
        print(
            "TRAINED_CPU_SMOKE_SUMMARY=" + json.dumps(status, sort_keys=True),
            flush=True,
        )
        return status
    except Exception as exc:
        atomic_json(
            out / "status.json",
            {
                "state": "failed",
                "stage": stage,
                "checked_comparisons": checked,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    smoke(parser.parse_args())


if __name__ == "__main__":
    main()
