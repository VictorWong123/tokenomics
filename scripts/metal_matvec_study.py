#!/usr/bin/env python3
"""Build three Metal kernel candidates and compare synthetic matrix components."""

import argparse
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.matvec_component import PROFILES, summarize
from bench.common import (
    ROOT,
    atomic_json,
    hardware,
    sha256,
    source_fingerprint,
    write_jsonl,
)
from scripts.diagnose_models import build_configuration
from scripts.metal_short_mv_study import command, tool


def study(args):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Run natively on the Apple Silicon Mac")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = source_fingerprint()
    report = {
        "state": "building",
        "source": source,
        "harness_source_sha256": sha256(ROOT / "tests/test_matvec_component.cpp"),
        "hardware": hardware(),
        "purpose": "Synthetic matrix components; no model downloads, trained-token or generation speed claim",
    }
    atomic_json(out / "manifest.json", report)
    try:
        binaries = {}
        for profile in PROFILES:
            build = ROOT / f"build-metal-{profile}"
            print(f"Matrix component: building {profile}; logs in {out}", flush=True)
            flags = {
                "CMAKE_BUILD_TYPE": "Release",
                "GGML_METAL": "ON",
                "GGML_METAL_EMBED_LIBRARY": "ON",
                "GGML_CUDA": "OFF",
                "GGML_NATIVE": "ON",
                "TOKENOMICS_SANITIZE": "OFF",
                "TOKENOMICS_VERIFY_STAGE_TIMING": "OFF",
                "TOKENOMICS_GREEDY_NO_TEMP": "OFF",
                "TOKENOMICS_CPU_BATCH_INVARIANT": "ON",
                "TOKENOMICS_CPU_PACKED_VERIFY": "OFF",
                "TOKENOMICS_CPU_Q4_PAIR": "OFF",
                "TOKENOMICS_METAL_SHORT_MV": "ON",
                "TOKENOMICS_METAL_INTERLEAVE": (
                    "ON" if profile == "interleaved" else "OFF"
                ),
                "TOKENOMICS_METAL_Q8_GROUP": "ON" if profile == "q8-group" else "OFF",
                "BUILD_TESTING": "ON",
                "Python3_EXECUTABLE": sys.executable,
            }
            command(
                [
                    tool("cmake"),
                    "-S",
                    ROOT,
                    "-B",
                    build,
                    *[f"-D{k}={v}" for k, v in flags.items()],
                ],
                out / f"{profile}-configure.log",
            )
            command(
                [
                    tool("cmake"),
                    "--build",
                    build,
                    "--target",
                    "test_matvec_component",
                    "-j2",
                ],
                out / f"{profile}-compile.log",
            )
            binaries[profile] = build / "test_matvec_component"
        report["binaries"] = {
            profile: {
                "sha256": sha256(binary),
                "configuration": build_configuration(binary),
            }
            for profile, binary in binaries.items()
        }
        report["state"] = "running"
        atomic_json(out / "manifest.json", report)

        def check_identities():
            if (
                source_fingerprint() != source
                or any(
                    sha256(binary) != report["binaries"][name]["sha256"]
                    for name, binary in binaries.items()
                )
                or sha256(ROOT / "tests/test_matvec_component.cpp")
                != report["harness_source_sha256"]
            ):
                raise RuntimeError(
                    "Source or executable changed during component study"
                )

        check_identities()
        records = []
        for block in range(3):
            order = PROFILES[block:] + PROFILES[:block]
            for profile in order:
                print(f"Matrix component: block{block + 1}/3 {profile}", flush=True)
                check_identities()
                raw = out / f"{profile}-{block}.jsonl"
                command([binaries[profile], "--metal"], raw)
                check_identities()
                # Backend diagnostics go to the same raw log. Only explicit JSON
                # lines are data; the original unfiltered file remains retained.
                rows = read_jsonl_lines(raw)
                if any(r["profile"] != profile for r in rows):
                    raise ValueError(
                        "Executable profile marker differs from requested build"
                    )
                records.extend(dict(row, block=block) for row in rows)
        summary = summarize(records)
        write_jsonl(out / "records.jsonl", records)
        atomic_json(out / "summary.json", summary)
        report["state"] = "passed"
        print(f"Matrix component complete: {out / 'summary.json'}", flush=True)
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        raise
    finally:
        atomic_json(out / "manifest.json", report)


def read_jsonl_lines(path):
    import json

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.startswith("{")
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    study(parser.parse_args())
