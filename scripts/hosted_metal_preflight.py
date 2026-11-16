"""Compile Apple profiles on a standard hosted runner; never benchmark throughput."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Requires native Apple Silicon macOS")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "purpose": "Apple compilation and correctness only; no M3 speed claim",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "synthetic_only": True,
        "stages": [],
    }

    def save():
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, command):
        log = out / f"{name}.log"
        with log.open("wb") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=stream)
        report["stages"].append(
            {
                "name": name,
                "command": [str(part) for part in command],
                "exit_code": result.returncode,
                "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            }
        )
        save()
        if result.returncode:
            # GitHub displays the failure without requiring an artifact download.
            print(log.read_text(errors="replace")[-18000:], flush=True)
            raise RuntimeError(f"{name} failed ({result.returncode})")

    try:
        run(
            "device-probe",
            [
                "swift",
                "-e",
                "import Metal; import Foundation; let d = MTLCreateSystemDefaultDevice(); "
                'let r: [String: Any] = ["available": d != nil, "name": d?.name ?? ""]; '
                "let b = try JSONSerialization.data(withJSONObject: r, options: [.sortedKeys]); "
                "print(String(data: b, encoding: .utf8)!)",
            ],
        )
        report["metal_device"] = json.loads((out / "device-probe.log").read_text())
        for profile, short in (("reference", "OFF"), ("short-mv", "ON")):
            build = ROOT / f"build-hosted-{profile}"
            run(
                f"{profile}-configure",
                [
                    sys.prefix + "/bin/cmake",
                    "-S",
                    str(ROOT),
                    "-B",
                    str(build),
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DCMAKE_MAKE_PROGRAM={sys.prefix}/bin/ninja",
                    f"-DPython3_EXECUTABLE={sys.executable}",
                    "-DBUILD_TESTING=ON",
                    "-DTOKENOMICS_PYTHON_TESTS=ON",
                    "-DGGML_NATIVE=ON",
                    "-DGGML_METAL=ON",
                    "-DGGML_METAL_EMBED_LIBRARY=ON",
                    "-DGGML_CUDA=OFF",
                    "-DTOKENOMICS_CPU_BATCH_INVARIANT=ON",
                    "-DTOKENOMICS_CPU_PACKED_VERIFY=OFF",
                    "-DTOKENOMICS_CPU_Q4_PAIR=OFF",
                    "-DTOKENOMICS_VERIFY_STAGE_TIMING=OFF",
                    "-DTOKENOMICS_GREEDY_NO_TEMP=OFF",
                    f"-DTOKENOMICS_METAL_SHORT_MV={short}",
                ],
            )
            run(
                f"{profile}-compile",
                [sys.prefix + "/bin/cmake", "--build", str(build), "-j", "2"],
            )
            run(
                f"{profile}-backend-compile",
                [
                    sys.prefix + "/bin/cmake",
                    "--build",
                    str(build),
                    "--target",
                    "test_backend_rows",
                    "-j",
                    "2",
                ],
            )
            report.setdefault("binary_sha256", {})[profile] = hashlib.sha256(
                (build / "tokenomics").read_bytes()
            ).hexdigest()
            run(
                f"{profile}-ctest",
                [
                    sys.prefix + "/bin/ctest",
                    "--test-dir",
                    str(build),
                    "--output-on-failure",
                    "--timeout",
                    "240",
                ],
            )
            run(f"{profile}-cpu-rows", [str(build / "test_backend_rows"), "--cpu"])
            if report["metal_device"]["available"]:
                run(
                    f"{profile}-metal-rows",
                    [str(build / "test_backend_rows"), "--metal"],
                )
        report["state"] = "passed"
        report["metal_execution"] = (
            "passed" if report["metal_device"]["available"] else "unavailable"
        )
    except Exception as exc:
        report["state"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
