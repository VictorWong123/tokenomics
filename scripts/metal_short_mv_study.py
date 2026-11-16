#!/usr/bin/env python3
"""Build and gate the opt-in Metal profile, then run a small paired local screen."""

import argparse
from functools import partial
from pathlib import Path
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.metal_short_mv import (
    SCREENS,
    analyze,
    kernel_variant,
    screen_configuration,
)
from bench.common import (
    ROOT,
    atomic_json,
    hardware,
    read_jsonl,
    sha256,
    source_fingerprint,
)
from bench.model_identity import DEFAULT_MODEL, MODEL_PINS, require_model_variant
from bench.run import run
from scripts.check_metal_profile import (
    direct_profile,
    gate,
    interleaved_profile,
    metal_profile,
    q8_group_profile,
    q8_shared_profile,
)
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
    model_id = getattr(args, "model_id", DEFAULT_MODEL)
    identity = require_model_variant(
        model_id,
        grouped_q8=getattr(args, "group_q8", False) or getattr(args, "share_q8", False),
    )
    if sha256(args.model) != identity["sha256"]:
        raise ValueError(
            f"Requires the pinned {model_id} Q4_K_M model; model identity differs"
        )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = source_fingerprint()
    atomic_json(out / "environment-before.json", hardware())
    atomic_json(out / "source.json", source)
    atomic_json(out / "model-identity.json", identity)
    interleaved = getattr(args, "interleave", False)
    group_q8 = getattr(args, "group_q8", False)
    shared_q8 = getattr(args, "share_q8", False)
    direct = getattr(args, "direct_greedy", False)
    candidate = kernel_variant(interleaved, group_q8, shared_q8, direct)
    profile_check = {
        "direct": direct_profile,
        "short-mv": metal_profile,
        "interleaved": interleaved_profile,
        "q8-group": q8_group_profile,
        "q8-shared": q8_shared_profile,
    }[candidate]
    if model_id != DEFAULT_MODEL:
        profile_check = partial(profile_check, model_id=model_id)
    builds = {
        "guarded": ROOT / "build-metal-reference",
        candidate: ROOT / f"build-metal-{candidate}",
    }
    state = "building"
    try:
        binaries = {}
        for profile, build in builds.items():
            enabled = "ON" if profile == candidate and not direct else "OFF"
            direct_flag = "ON" if profile == candidate and direct else "OFF"
            locality = "ON" if interleaved and profile == candidate else "OFF"
            grouped = (
                "ON" if (group_q8 or shared_q8) and profile == candidate else "OFF"
            )
            shared = "ON" if shared_q8 and profile == candidate else "OFF"
            print(f"Metal study: configuring {profile}; logs in {out}", flush=True)
            command(
                [
                    tool("cmake"),
                    "-S",
                    ROOT,
                    "-B",
                    build,
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DTOKENOMICS_SANITIZE=OFF",
                    "-DTOKENOMICS_VERIFY_STAGE_TIMING=OFF",
                    f"-DTOKENOMICS_GREEDY_NO_TEMP={direct_flag}",
                    f"-DTOKENOMICS_GREEDY_DIRECT={direct_flag}",
                    f"-DTOKENOMICS_METAL_INTERLEAVE={locality}",
                    f"-DTOKENOMICS_METAL_Q8_GROUP={grouped}",
                    f"-DTOKENOMICS_METAL_Q8_SHARED={shared}",
                    "-DGGML_METAL=ON",
                    "-DGGML_METAL_EMBED_LIBRARY=ON",
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
        if direct:
            state = "host_sampler_equivalence"
            command(
                [builds[candidate] / "test_greedy_equivalence"],
                out / "sampler-equivalence.log",
            )
            atomic_json(
                out / "sampler-checks.json",
                {
                    "state": "passed",
                    "scope": "Host strict sampler equivalence; trained Metal gate follows",
                    "binary_sha256": sha256(
                        builds[candidate] / "test_greedy_equivalence"
                    ),
                    "log_sha256": sha256(out / "sampler-equivalence.log"),
                },
            )
        else:
            state = "direct_backend_rows"
            candidate_build = builds[candidate]
            command(
                [
                    tool("cmake"),
                    "--build",
                    candidate_build,
                    "--target",
                    "test_backend_rows",
                    "-j2",
                ],
                out / "backend-rows-compile.log",
            )
            command(
                [candidate_build / "test_backend_rows", "--metal"],
                out / "backend-rows.log",
            )
            atomic_json(
                out / "backend-rows.json",
                {
                    "state": "passed",
                    "backend": "Metal",
                    "binary_sha256": sha256(candidate_build / "test_backend_rows"),
                    "log_sha256": sha256(out / "backend-rows.log"),
                    "limitation": "Matrix row arithmetic only; trained exactness gate still required",
                },
            )
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
            binaries[candidate],
            args.model.resolve(),
            out / "gate",
            profile_validator=profile_check,
            model_id=model_id,
        )
        check_binaries()
        screen = getattr(args, "screen", "ngram")
        if screen == "retrieval":
            state = "suffix_correctness_gate"
            gate(
                binaries["guarded"],
                binaries[candidate],
                args.model.resolve(),
                out / "gate-suffix",
                drafter="suffix",
                profile_validator=profile_check,
                model_id=model_id,
            )
            check_binaries()
        if args.gate_only:
            state = "gate_passed_no_timings"
        else:
            state = "development_screen"
            config = screen_configuration(
                screen, interleaved, group_q8, shared_q8, direct, model_id
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
                f"Metal study: exact gates passed; running {len(config['configurations']) * 9} paired development records",
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
                identity={"purpose": f"local M3 {candidate} {screen} component screen"},
                executables=executables,
            )
            analyze(
                out / "screen",
                screen=screen,
                interleaved=interleaved,
                group_q8=group_q8,
                shared_q8=shared_q8,
                direct=direct,
                model_id=model_id,
            )
            state = "complete"
        if source_fingerprint() != source:
            raise RuntimeError("Source changed during the study")
        check_binaries()
        atomic_json(
            out / "status.json",
            {
                "state": state,
                "gate": "passed",
                "screen": screen,
                "model_id": model_id,
                "kernel_variant": "guarded-single-row" if direct else candidate,
                "sampler_variant": "direct" if direct else "original",
                "gate_drafters": (
                    ["ngram", "suffix"] if screen == "retrieval" else ["ngram"]
                ),
                "timing_evidence": not args.gate_only,
            },
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
    locality = parser.add_mutually_exclusive_group()
    locality.add_argument(
        "--direct-greedy",
        action="store_true",
        help="Isolate strict direct-logit target sampling with guarded one-row Metal verification",
    )
    locality.add_argument(
        "--share-q8",
        action="store_true",
        help="Share Q8 weight loads across two query accumulators per threadgroup",
    )
    locality.add_argument(
        "--group-q8",
        action="store_true",
        help="Run two Q8 query rows per threadgroup using the unchanged arithmetic routine",
    )
    locality.add_argument(
        "--interleave",
        action="store_true",
        help="Build/gate the weight-tile interleaving candidate instead of plain short-MV",
    )
    parser.add_argument(
        "--screen",
        choices=tuple(SCREENS),
        default="ngram",
        help="Retrieval adds suffix caps1/4/8 and a separate suffix correctness gate",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", choices=tuple(MODEL_PINS), default=DEFAULT_MODEL)
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
