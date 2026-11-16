#!/usr/bin/env python3
"""Local M3 suffix gate and paired development screen with ordinary/ngram controls."""

import argparse
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.suffix_study import analyze
from bench.common import (
    ROOT,
    atomic_json,
    hardware,
    read_jsonl,
    sha256,
    source_fingerprint,
)
from bench.run import run
from scripts.check_metal_profile import MODEL_SHA256, gate, metal_profile
from scripts.diagnose_models import build_configuration


def study(args):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Run this study natively on Apple Silicon")
    if sha256(args.model) != MODEL_SHA256:
        raise ValueError("Expected the pinned Qwen2.5 0.5B Q4_K_M model")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary, model = args.binary.resolve(), args.model.resolve()
    source = source_fingerprint()
    binary_hash = sha256(binary)
    atomic_json(out / "environment-before.json", hardware())
    atomic_json(
        out / "compilation.json",
        {
            "source": source,
            "configuration": build_configuration(binary),
            "binary_sha256": binary_hash,
        },
    )
    stage = "correctness_gate"
    try:
        print(
            "Suffix study: exact gate against ordinary decode, including M3 regression",
            flush=True,
        )
        # Both mechanisms use the same guarded binary; this isolates proposals.
        gate(
            binary,
            binary,
            model,
            out / "gate",
            drafter="suffix",
            profile_validator=lambda row, candidate: metal_profile(row, False),
        )
        if sha256(binary) != binary_hash or source_fingerprint() != source:
            raise RuntimeError("Executable or source changed during the gate")
        if args.gate_only:
            stage = "gate_passed_no_timings"
        else:
            stage = "development_screen"
            config = json.loads((ROOT / "configs/suffix-screen-v1.json").read_text())
            prompts = [
                p
                for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")
                if p["id"] in config["prompt_ids"]
            ]
            print(
                "Suffix study: gate passed; running 72 paired development records",
                flush=True,
            )
            run(
                binary,
                model,
                None,
                prompts,
                config,
                out / "screen",
                gpu_layers=99,
                identity={
                    "purpose": "suffix minimum5 cap1 development screen; threshold selected on these fit prompts"
                },
            )
            analyze(out / "screen")
            stage = "complete"
        if sha256(binary) != binary_hash or source_fingerprint() != source:
            raise RuntimeError("Executable or source changed during the study")
        atomic_json(
            out / "status.json",
            {"state": stage, "gate": "passed", "timing_evidence": not args.gate_only},
        )
        print(f"Suffix study: {stage}; evidence: {out}", flush=True)
    except Exception as exc:
        atomic_json(
            out / "status.json",
            {
                "state": "failed",
                "stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        atomic_json(out / "environment-after.json", hardware())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build-metal/tokenomics")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate-only", action="store_true")
    study(parser.parse_args())


if __name__ == "__main__":
    main()
