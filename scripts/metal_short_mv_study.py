#!/usr/bin/env python3
"""Build and gate the opt-in Metal profile, then run a small paired local screen."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.metal_short_mv import analyze
from bench.common import (
    ROOT,
    atomic_json,
    hardware,
    read_jsonl,
    sha256,
    source_fingerprint,
)
from bench.run import run
from scripts.check_metal_profile import MODEL_SHA256, gate
from scripts.diagnose_models import build_configuration


def tool(name):
    local = Path(sys.executable).with_name(name)
    found = str(local) if local.is_file() else shutil.which(name)
    if not found:
        raise RuntimeError(f"Missing {name}; install requirements.txt first")
    return found


def command(args, log):
    with log.open("w") as stream:
        subprocess.run(
            [str(arg) for arg in args],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def study(args):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "Run this study natively on the Apple Silicon Mac, not Linux or Rosetta"
        )
    if sha256(args.model) != MODEL_SHA256:
        raise ValueError(
            "Requires the pinned Qwen2.5 0.5B Q4_K_M model; model identity differs"
        )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = source_fingerprint()
    atomic_json(out / "environment-before.json", hardware())
    atomic_json(out / "source.json", source)
    builds = {
        "guarded": ROOT / "build-metal-reference",
        "short-mv": ROOT / "build-metal-short-mv",
    }
    state = "building"
    try:
        binaries = {}
        for profile, build in builds.items():
            enabled = "ON" if profile == "short-mv" else "OFF"
            print(f"Metal study: configuring {profile}; logs in {out}", flush=True)
            command(
                [
                    tool("cmake"),
                    "-S",
                    ROOT,
                    "-B",
                    build,
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_METAL=ON",
                    "-DGGML_CUDA=OFF",
                    "-DGGML_NATIVE=ON",
                    "-DTOKENOMICS_CPU_BATCH_INVARIANT=ON",
                    "-DTOKENOMICS_CPU_PACKED_VERIFY=OFF",
                    "-DTOKENOMICS_CPU_Q4_PAIR=OFF",
                    f"-DTOKENOMICS_METAL_SHORT_MV={enabled}",
                    f"-DPython3_EXECUTABLE={sys.executable}",
                    "-DBUILD_TESTING=ON",
                    "-DTOKENOMICS_PYTHON_TESTS=ON",
                ],
                out / f"{profile}-configure.log",
            )
            command(
                [tool("cmake"), "--build", build, "-j2"], out / f"{profile}-compile.log"
            )
            command(
                [tool("ctest"), "--test-dir", build, "--output-on-failure"],
                out / f"{profile}-tests.log",
            )
            binaries[profile] = build / "tokenomics"
        atomic_json(
            out / "compilation.json",
            {
                profile: {
                    "configuration": build_configuration(binary),
                    "binary_sha256": sha256(binary),
                    "generated_source_sha256": {
                        p.name: sha256(p)
                        for p in binary.parent.glob("tokenomics-*.cpp")
                    },
                }
                for profile, binary in binaries.items()
            },
        )
        if source_fingerprint() != source:
            raise RuntimeError("Source changed while compiling")
        binary_hashes = {
            profile: sha256(binary) for profile, binary in binaries.items()
        }

        def check_binaries():
            if any(
                sha256(binary) != binary_hashes[profile]
                for profile, binary in binaries.items()
            ):
                raise RuntimeError(
                    "Binary changed after compilation; gate identity is invalid"
                )

        state = "correctness_gate"
        print(
            "Metal study: original M3 regression, then cross-profile exactness gate",
            flush=True,
        )
        gate(
            binaries["guarded"],
            binaries["short-mv"],
            args.model.resolve(),
            out / "gate",
        )
        check_binaries()
        if args.gate_only:
            state = "gate_passed_no_timings"
        else:
            state = "development_screen"
            config = json.loads(
                (ROOT / "configs/metal-short-mv-screen-v1.json").read_text()
            )
            prompts = [
                p
                for p in read_jsonl(ROOT / "bench/prompts-v1.jsonl")
                if p["id"] in config["prompt_ids"]
            ]
            executables = {
                c["id"]: binaries[c["kernel_profile"]] for c in config["configurations"]
            }
            print(
                "Metal study: exact gate passed; running 72 paired development records",
                flush=True,
            )
            run(
                binaries["guarded"],
                args.model.resolve(),
                None,
                prompts,
                config,
                out / "screen",
                gpu_layers=99,
                identity={"purpose": "local M3 short-MV component screen"},
                executables=executables,
            )
            analyze(out / "screen")
            state = "complete"
        if source_fingerprint() != source:
            raise RuntimeError("Source changed during the study")
        check_binaries()
        atomic_json(
            out / "status.json",
            {"state": state, "gate": "passed", "timing_evidence": not args.gate_only},
        )
        print(f"Metal study: {state}; evidence: {out}", flush=True)
    except Exception as exc:
        atomic_json(
            out / "status.json",
            {
                "state": "failed",
                "stage": state,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        raise
    finally:
        atomic_json(out / "environment-after.json", hardware())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Fresh output directory; failed runs are never overwritten",
    )
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="Build and check correctness without running timings",
    )
    args = parser.parse_args()
    study(args)


if __name__ == "__main__":
    main()
