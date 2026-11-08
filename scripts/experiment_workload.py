#!/usr/bin/env python3
"""Fixed development experiments using the existing owned Codespace lifecycle."""
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import atomic_json, read_jsonl, sha256
from scripts.codespaces import controller as lifecycle
from scripts.model_workload import CACHE

SUFFIXES = {".json", ".jsonl", ".txt", ".log", ".png", ".md"}


def pack(results, destination, owner):
    """Bounded experiment artifacts only; never archive a workspace or cache."""
    entries = []
    total = 0
    for source in sorted(results.rglob("*")):
        lifecycle.require(not source.is_symlink(), "Symlink in experiment evidence")
        if not source.is_file():
            continue
        lifecycle.require(
            (
                source.suffix in SUFFIXES
                or source.name.endswith((".json.tmp", ".jsonl.tmp"))
            )
            and source.resolve().is_relative_to(results.resolve()),
            "Unexpected experiment artifact",
        )
        size = source.stat().st_size
        total += size
        lifecycle.require(
            size <= 32 * 1024**2 and total <= 512 * 1024**2,
            "Experiment evidence size limit",
        )
        entries.append((source.relative_to(results).as_posix(), source, size))
    lifecycle.require(
        any(n == "status.json" for n, _, _ in entries), "Missing stage status"
    )
    manifest = {
        **owner,
        "files": {n: {"sha256": sha256(p), "bytes": s} for n, p, s in entries},
    }
    raw = json.dumps(manifest, sort_keys=True).encode()
    with tarfile.open(destination, "w:gz") as archive:
        for name, source, _ in entries:
            archive.add(source, arcname=name, recursive=False)
        member = tarfile.TarInfo("execution-manifest.json")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    return sha256(destination)


def workload():
    phase = os.environ.get("TOKENOMICS_EXPERIMENT_PHASE", "timing")
    lifecycle.require(
        phase
        in (
            "timing",
            "pilot",
            "confirmation",
            "kernel",
            "controller",
            "controller-packed",
            "profiles",
        ),
        "Unsupported fixed experiment",
    )
    selected = os.environ.get("TOKENOMICS_EXPERIMENT_CANDIDATE", "all")
    lifecycle.require(
        selected in ("all", "qwen25-05-15-q4", "qwen25-05-3-q4"),
        "Unsupported fixed candidate",
    )
    candidate = "qwen25-05-15-q4" if selected == "all" else selected
    drafter = os.environ.get("TOKENOMICS_EXPERIMENT_DRAFTER", "model")
    lifecycle.require(drafter in ("model", "ngram"), "Unsupported fixed drafter")
    lifecycle.require(
        phase
        not in ("confirmation", "kernel", "controller", "controller-packed", "profiles")
        or (selected == "qwen25-05-3-q4" and drafter == "ngram"),
        "Confirmation/kernel/controller/profile experiments require the 3B target and ngram drafter",
    )
    point = candidate + ("-ngram" if drafter == "ngram" else "")
    owner = lifecycle.current()
    expected = lifecycle.expected_runner()
    root = lifecycle.ownership(owner)
    lifecycle.registered_runner(expected)
    lifecycle.require(os.environ.get("RUNNER_NAME") == expected["name"], "Wrong runner")
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    user_checkout = Path("/workspaces/tokenomics")
    lifecycle.require(
        not workspace.is_relative_to(user_checkout)
        and not user_checkout.is_relative_to(workspace),
        "Unsafe workspace",
    )
    lifecycle.require(
        lifecycle.command(["git", "-C", str(workspace), "rev-parse", "HEAD"])
        == owner["sha"],
        "Checkout identity mismatch",
    )
    if phase in ("pilot", "confirmation", "controller", "controller-packed"):
        # Retain only experiment measurements across owned stop/start cycles.
        # Lifecycle leases, runner configuration and credentials stay elsewhere.
        store = Path("/workspaces/.tokenomics-experiment-cache")
        lifecycle.require(not store.is_symlink(), "Unsafe experiment cache")
        store.mkdir(mode=0o700, exist_ok=True)
        lifecycle.mode_owner(store.lstat(), 0o700, directory=True)
        results = store / owner["sha"]
        lifecycle.require(not results.is_symlink(), "Unsafe experiment directory")
        results.mkdir(mode=0o700, exist_ok=True)
        lifecycle.mode_owner(results.lstat(), 0o700, directory=True)
    else:
        results = root / "experiment-results"
        results.mkdir(exist_ok=False)
    artifacts = (
        Path(os.environ["RUNNER_TEMP"])
        / f"tokenomics-experiment-{owner['run_id']}-{owner['attempt']}"
    )
    artifacts.mkdir(exist_ok=False)
    status = {"schema_version": 1, "state": "started", "stage": "bootstrap"}
    atomic_json(results / "status.json", status)
    env = lifecycle.fixture_environment()
    code = 1
    try:
        print("TOKENOMICS_EXPERIMENT_STAGE bootstrap", flush=True)
        subprocess.run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=15s",
                "1800s",
                "bash",
                "scripts/bootstrap.sh",
                "cpu",
            ],
            cwd=workspace,
            env=env,
            check=True,
            timeout=1830,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        lifecycle.require(
            CACHE.is_dir() and not CACHE.is_symlink(), "Missing model cache"
        )
        lifecycle.mode_owner(CACHE.lstat(), 0o700, directory=True)
        for path in CACHE.iterdir():
            lifecycle.require(not path.is_symlink(), "Unsafe cache entry")
        status["stage"] = "development-" + phase
        atomic_json(results / "status.json", status)
        print("TOKENOMICS_EXPERIMENT_STAGE " + phase, flush=True)
        args = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            "1800s" if phase == "timing" else "9000s",
            str(workspace / ".venv/bin/python"),
            "scripts/run_all.py",
            "--backend",
            "cpu",
            "--threads",
            "2",
            "--model-dir",
            str(CACHE),
            "--run-dir",
            str(results),
        ]
        args += (
            ["--timing-only", "--candidate", candidate]
            if phase == "timing"
            else ["--budget-seconds", "6600"]
        )
        if phase == "pilot" and selected != "all":
            args += ["--candidate", candidate]
        if drafter == "ngram":
            args += ["--ngram-only"]
        if phase == "confirmation":
            args = [
                "timeout",
                "--signal=TERM",
                "--kill-after=15s",
                "9000s",
                str(workspace / ".venv/bin/python"),
                "scripts/confirm_ngram.py",
                "--model-dir",
                str(CACHE),
                "--run-dir",
                str(results),
                "--budget-seconds",
                "6600",
            ]
        elif phase in ("kernel", "profiles"):
            args = [
                "timeout",
                "--signal=TERM",
                "--kill-after=15s",
                "9000s",
                str(workspace / ".venv/bin/python"),
                (
                    "scripts/check_kernel_profile.py"
                    if phase == "kernel"
                    else "scripts/compare_cpu_profiles.py"
                ),
                "--model-dir",
                str(CACHE),
                "--run-dir",
                str(results),
            ]
        elif phase in ("controller", "controller-packed"):
            args = [
                "timeout",
                "--signal=TERM",
                "--kill-after=15s",
                "9000s",
                str(workspace / ".venv/bin/python"),
                "scripts/controller_study.py",
                "--model-dir",
                str(CACHE),
                "--run-dir",
                str(results),
                "--budget-seconds",
                "6600",
            ]
            if phase == "controller-packed":
                args += ["--cpu-profile", "packed"]
        with open(
            results / f"pipeline-{owner['run_id']}-{owner['attempt']}.txt", "x"
        ) as log:
            code = subprocess.run(
                args,
                cwd=workspace,
                env=env,
                timeout=1830 if phase == "timing" else 9030,
                stdout=log,
                stderr=log,
            ).returncode
        status = json.loads((results / "status.json").read_text())
        expected_states = {
            "confirmation": {0: "confirmation_complete", 11: "paused"},
            "kernel": {0: "kernel_profile_passed"},
            "controller": {0: "controller_study_complete", 11: "paused"},
            "controller-packed": {0: "controller_study_complete", 11: "paused"},
            "profiles": {0: "profile_comparison_complete"},
            "timing": {10: "timing_complete"},
            "pilot": {
                11: "paused",
                20: "proceed_controller_implementation",
                21: "narrow_or_investigate",
                22: "scope_decision_required",
            },
        }[phase]
        lifecycle.require(
            code in expected_states and status["state"] == expected_states[code],
            "Experiment did not satisfy its result contract",
        )
        timing = (
            results / "profiles"
            if phase == "profiles"
            else (
                results / "validation"
                if phase == "controller"
                else (
                    results / "confirmation"
                    if phase == "confirmation"
                    else results / point / "timing"
                )
            )
        )
        if phase == "controller" and status["state"] == "paused":
            rows = (
                read_jsonl(timing / "requests.jsonl")
                if (timing / "requests.jsonl").exists()
                else []
            )
            checks = (
                read_jsonl(timing / "correctness.jsonl")
                if (timing / "correctness.jsonl").exists()
                else []
            )
        else:
            rows = read_jsonl(timing / "requests.jsonl")
            checks = read_jsonl(timing / "correctness.jsonl")
        lifecycle.require(
            (
                len(rows) % 3 == 0
                and len(rows) <= 162
                and (status["state"] != "confirmation_complete" or len(rows) == 162)
                if phase == "confirmation"
                else (
                    (
                        len(rows) % 7 == 0
                        and len(rows) <= 252
                        and (
                            status["state"] != "controller_study_complete"
                            or len(rows) == 252
                        )
                    )
                    if phase == "controller"
                    else len(rows)
                    == (
                        54
                        if phase == "profiles"
                        else (18 if drafter == "ngram" else 21)
                    )
                )
            )
            and len(checks) == len(rows)
            and all(c.get("ok") is True for c in checks),
            "Invalid timing evidence",
        )
        if phase in ("timing", "kernel"):
            status["stage"] = "checkpoint-validation"
            atomic_json(results / "status.json", status)
            print("TOKENOMICS_EXPERIMENT_STAGE checkpoint-validation", flush=True)
            catalog = json.loads((workspace / "configs/models-v1.json").read_text())
            pair = next(c for c in catalog["candidates"] if c["id"] == candidate)
            target = CACHE / catalog["models"][pair["target"]]["file"]
            validation = results / "checkpoint-validation"
            with open(results / "checkpoint-validation.txt", "x") as log:
                replay_args = [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=15s",
                    "1800s",
                    str(workspace / ".venv/bin/python"),
                    "-m",
                    "bench.replay",
                    "--exe",
                    str(workspace / "build-cpu/tokenomics"),
                    "--target",
                    str(target),
                    "--pilot-dir",
                    str(timing),
                    "--out",
                    str(validation),
                ]
                if drafter == "model":
                    replay_args += [
                        "--draft",
                        str(CACHE / catalog["models"][pair["draft"]]["file"]),
                    ]
                subprocess.run(
                    replay_args,
                    cwd=workspace,
                    env=env,
                    timeout=1830,
                    check=True,
                    stdout=log,
                    stderr=log,
                )
            groups = len(list(validation.glob("*/checkpoint-*/complete.json")))
            lifecycle.require(
                groups > 0, "No checkpoint path exercised on trained weights"
            )
            status["checkpoint_validation"] = {
                "state": "passed",
                "groups": groups,
                "protocol": json.loads((validation / "decision.json").read_text())[
                    "state_protocol"
                ],
            }
    except Exception as exc:
        status.update(state="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        atomic_json(results / "status.json", status)
        lifecycle.ownership(owner)
        digest = pack(results, artifacts / "evidence.tar.gz", owner)
        report = {
            **owner,
            "workload": "development-" + phase,
            "candidate": candidate if phase == "timing" else selected,
            "drafter": drafter,
            "hostname": socket.gethostname(),
            "runner_id": expected["id"],
            "state": status["state"],
            "archive_sha256": digest,
            "status": status,
        }
        if status["state"] != "failed":
            report["measurements"] = [
                {
                    k: r[k]
                    for k in (
                        "prompt_id",
                        "configuration_id",
                        "generated_tokens",
                        "generation_ms",
                        "warm_ms",
                        "peak_rss_bytes",
                        "correctness",
                    )
                }
                for r in rows
            ]
            report["environment"] = json.loads(
                (results / "environment.json").read_text()
            )
            report["completed_pilot_blocks"] = len(
                list(results.glob("*/pilot-a/blocks/*/complete.json"))
            )
            report["completed_confirmation_blocks"] = len(
                list(results.glob("confirmation/blocks/*/complete.json"))
            )
            for label, path in (
                ("pilot_a", results / point / "pilot-a/analysis/summary.json"),
                ("pilot_b", results / point / "pilot-b/decision.json"),
                ("confirmation", results / "confirmation/analysis/summary.json"),
                ("controller", results / "validation/analysis/summary.json"),
                ("controller_logging", results / "logging/analysis/summary.json"),
                ("kernel_gate", results / "kernel-gate/status.json"),
                ("kernel_screen", results / point / "analysis/measurements.json"),
                ("cpu_profiles", results / "profiles/analysis/summary.json"),
            ):
                if path.exists():
                    report[label] = json.loads(path.read_text())
        atomic_json(artifacts / "report.json", report)
        print(
            "TOKENOMICS_EXPERIMENT_SUMMARY " + json.dumps(report, sort_keys=True),
            flush=True,
        )
    return (
        0
        if status["state"]
        in (
            "timing_complete",
            "confirmation_complete",
            "kernel_profile_passed",
            "controller_study_complete",
            "profile_comparison_complete",
            "paused",
            "proceed_controller_implementation",
            "narrow_or_investigate",
            "scope_decision_required",
        )
        else 1
    )


if __name__ == "__main__":
    if len(sys.argv) != 1:
        sys.exit("Fixed workload only")
    sys.exit(workload())
