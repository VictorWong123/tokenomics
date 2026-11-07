#!/usr/bin/env python3
"""Fixed Linux model verification workload for the verified Codespace lifecycle.

No lifecycle token, arbitrary command input, or resource creation. Initialize
and finalize remain owned by scripts/codespaces/controller.py.
"""
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.codespaces import controller as lifecycle
from bench.common import atomic_json, available_memory, hardware, sha256

CACHE = Path("/workspaces/.tokenomics-model-cache")
GATE_FILES = ("manifest.json", "status.json", "correctness.jsonl") + tuple(
    name + suffix
    for name in ("reference", "boundary-reference", "candidate")
    for suffix in (".jobs.jsonl", ".jsonl", ".cycles.jsonl", ".stderr.log")
)
DIAGNOSTIC_FILES = ("manifest.json", "status.json") + tuple(
    name + suffix
    for name in ("reference", "full", "replay")
    for suffix in (".jobs.jsonl", ".jsonl", ".cycles.jsonl", ".stderr.log")
)
FILES = (
    ("status.json", "environment.json", "models.lock.json")
    + tuple("gate/" + name for name in GATE_FILES)
    + tuple("diagnostic/" + name for name in DIAGNOSTIC_FILES)
)
MAX_FILE = 16 * 1024**2
MAX_TOTAL = 48 * 1024**2


def pack(results, destination, owner):
    """Only enumerated, regular non-symlink files; include failed partial attempts."""
    entries = []
    total = 0
    for name in FILES:
        source = results / name
        if not source.exists():
            continue
        lifecycle.require(
            source.is_file()
            and not source.is_symlink()
            and source.resolve().is_relative_to(results.resolve()),
            "Unsafe model evidence",
        )
        size = source.stat().st_size
        total += size
        lifecycle.require(
            size <= MAX_FILE and total <= MAX_TOTAL, "Model evidence size limit"
        )
        entries.append((name, source, size))
    lifecycle.require(
        any(name == "status.json" for name, _, _ in entries),
        "Missing model stage status",
    )
    status = json.loads((results / "status.json").read_text())
    if status.get("state") == "passed":
        lifecycle.require(
            {
                "status.json",
                "environment.json",
                "models.lock.json",
                *("gate/" + name for name in GATE_FILES),
            }.issubset({name for name, _, _ in entries}),
            "Incomplete successful model evidence",
        )
        checks = [
            json.loads(line)
            for line in (results / "gate/correctness.jsonl").read_text().splitlines()
        ]
        gate = json.loads((results / "gate/status.json").read_text())
        lifecycle.require(
            len(checks) == 272
            and all(row.get("ok") is True for row in checks)
            and gate.get("state") == "passed"
            and gate.get("synthetic_fixture") is False,
            "Invalid model correctness evidence",
        )
    if status.get("state") == "diagnostic_complete":
        lifecycle.require(
            {"diagnostic/" + name for name in DIAGNOSTIC_FILES}.issubset(
                {name for name, _, _ in entries}
            ),
            "Incomplete diagnostic evidence",
        )
        diagnostic = json.loads((results / "diagnostic/status.json").read_text())
        lifecycle.require(
            diagnostic.get("state") == "diagnostic_complete"
            and diagnostic.get("synthetic_fixture") is False
            and diagnostic.get("correctness_supported") is False
            and len(diagnostic.get("full_comparisons", [])) == 14,
            "Invalid diagnostic evidence",
        )
    manifest = {
        **owner,
        "hostname": socket.gethostname(),
        "files": {
            name: {"sha256": sha256(source), "bytes": size}
            for name, source, size in entries
        },
    }
    raw = json.dumps(manifest, sort_keys=True).encode()
    with tarfile.open(destination, "w:gz") as archive:
        for name, source, _ in entries:
            archive.add(source, arcname=name, recursive=False)
        member = tarfile.TarInfo("execution-manifest.json")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    return sha256(destination)


def child(results):
    from scripts.download_models import download
    from scripts.diagnose_models import diagnose
    from scripts.check_models import parser, run as check_models

    status = {"schema_version": 1, "state": "started", "stage": "resource-screen"}
    atomic_json(results / "environment.json", hardware())
    atomic_json(results / "status.json", status)
    print("TOKENOMICS_MODEL_STAGE resource-screen", flush=True)
    try:
        free = available_memory()
        status["available_bytes"] = free
        status["disk_free_bytes"] = shutil.disk_usage(CACHE.parent).free
        lifecycle.require(
            free >= 5 * 2**30, "Less than 5 GiB available for initial candidate"
        )
        lifecycle.require(
            status["disk_free_bytes"] >= 4 * 2**30,
            "Less than 4 GiB free disk",
        )
        CACHE.mkdir(mode=0o700, exist_ok=True)
        lifecycle.mode_owner(CACHE.lstat(), 0o700, directory=True)
        lock = CACHE / "models.lock.json"
        # Do not follow cache-file symlinks, including partial downloads.
        for path in CACHE.iterdir():
            lifecycle.require(not path.is_symlink(), "Unsafe model cache entry")
        status["stage"] = "model-download"
        atomic_json(results / "status.json", status)
        print("TOKENOMICS_MODEL_STAGE model-download", flush=True)
        draft = download("qwen25-05", CACHE, lock)
        target = download("qwen25-15", CACHE, lock)
        shutil.copyfile(lock, results / "models.lock.json")
        locked = json.loads(lock.read_text())["models"]
        status["target_sha256"] = locked["qwen25-15"]["sha256"]
        status["draft_sha256"] = locked["qwen25-05"]["sha256"]
        status["stage"] = "divergence-diagnosis"
        atomic_json(results / "status.json", status)
        print("TOKENOMICS_MODEL_STAGE divergence-diagnosis", flush=True)
        diagnostic = diagnose(target, draft, results / "diagnostic", threads=2)
        lifecycle.require(
            diagnostic["state"] == "diagnostic_complete"
            and diagnostic["synthetic_fixture"] is False
            and diagnostic["correctness_supported"] is False,
            "Diagnostic contract failed",
        )
        status["diagnosis"] = diagnostic
        lifecycle.require(
            diagnostic["reference_reset_equal"] is True
            and all(row["mismatch"] is None for row in diagnostic["full_comparisons"])
            and all(
                row["probe_matches_emitted"] for row in diagnostic["replay_probes"]
            ),
            "CPU diagnostic still diverges",
        )
        status["stage"] = "correctness"
        atomic_json(results / "status.json", status)
        print("TOKENOMICS_MODEL_STAGE correctness", flush=True)
        gate = check_models(
            parser().parse_args(
                [
                    "--target",
                    str(target),
                    "--draft",
                    str(draft),
                    "--out",
                    str(results / "gate"),
                    "--threads",
                    "2",
                    "--timeout-seconds",
                    "3600",
                ]
            )
        )
        lifecycle.require(
            gate["state"] == "passed"
            and gate["comparisons"] == 272
            and gate["synthetic_fixture"] is False,
            "Trained-model correctness contract failed",
        )
        status.update(
            state="passed",
            comparisons=gate["comparisons"],
            observed_coverage=gate["observed_coverage"],
        )
    except Exception as exc:
        status.update(state="failed", error_type=type(exc).__name__, error=str(exc))
        gate_status = results / "gate/status.json"
        if gate_status.exists():
            diagnostic = json.loads(gate_status.read_text())
            status["checked_comparisons"] = diagnostic.get("checked_comparisons", 0)
            status["first_mismatch"] = diagnostic.get("first_mismatch")
    atomic_json(results / "status.json", status)
    print("TOKENOMICS_MODEL_STAGE " + status["state"], flush=True)
    return 0 if status["state"] == "passed" else 1


def workload():
    owner = lifecycle.current()
    expected = lifecycle.expected_runner()
    root = lifecycle.ownership(owner)
    lifecycle.registered_runner(expected)
    lifecycle.require(
        os.environ.get("RUNNER_NAME") == expected["name"], "Wrong model runner"
    )
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    user_checkout = Path("/workspaces/tokenomics")
    lifecycle.require(
        not workspace.is_relative_to(user_checkout)
        and not user_checkout.is_relative_to(workspace),
        "User checkout cannot be model workspace",
    )
    lifecycle.require(
        lifecycle.command(["git", "-C", str(workspace), "rev-parse", "HEAD"])
        == owner["sha"],
        "Model checkout SHA mismatch",
    )
    results = root / "model-results"
    results.mkdir(exist_ok=False)
    artifacts = (
        Path(os.environ["RUNNER_TEMP"])
        / f"tokenomics-model-{owner['run_id']}-{owner['attempt']}"
    )
    artifacts.mkdir(exist_ok=False)
    env = lifecycle.fixture_environment()
    status = {"schema_version": 1, "state": "started", "stage": "bootstrap"}
    atomic_json(results / "status.json", status)
    print("TOKENOMICS_MODEL_STAGE bootstrap", flush=True)
    code = 1
    try:
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
        code = subprocess.run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=15s",
                "9000s",
                str(workspace / ".venv/bin/python"),
                str(workspace / "scripts/model_workload.py"),
                "--owned-child",
            ],
            cwd=workspace,
            env=env,
            timeout=9030,
            stdout=None,  # child emits only fixed stage names; engine output is captured
            stderr=subprocess.DEVNULL,
        ).returncode
        status = json.loads((results / "status.json").read_text())
        if code != 0 and status["state"] == "started":
            status.update(
                state="failed", error="Child interrupted or timed out", returncode=code
            )
    except Exception as exc:
        status.update(state="failed", error_type=type(exc).__name__)
    finally:
        atomic_json(results / "status.json", status)
        lifecycle.ownership(owner)
        digest = pack(results, artifacts / "evidence.tar.gz", owner)
        # No raw model output, prompts, credentials or network errors in Actions logs.
        report = {
            **owner,
            "workload": "trained-model-correctness",
            "hostname": socket.gethostname(),
            "runner_name": expected["name"],
            "runner_id": expected["id"],
            "execution_environment": (
                "codespaces" if os.environ.get("CODESPACES") == "true" else "other"
            ),
            "diagnosis": status.get("diagnosis"),
            "state": status["state"],
            "stage": status["stage"],
            "archive_sha256": digest,
            "comparisons": status.get("comparisons", 0),
            "observed_coverage": status.get("observed_coverage", {}),
            "error_type": status.get("error_type"),
            "checked_comparisons": status.get(
                "checked_comparisons", status.get("comparisons", 0)
            ),
            "first_mismatch": status.get("first_mismatch"),
            "available_bytes": status.get("available_bytes"),
            "disk_free_bytes": status.get("disk_free_bytes"),
            "target_sha256": status.get("target_sha256"),
            "draft_sha256": status.get("draft_sha256"),
        }
        atomic_json(artifacts / "report.json", report)
        failure = status.get("first_mismatch")
        if failure is None:
            failure = next(
                (
                    row
                    for row in status.get("diagnosis", {}).get("full_comparisons", [])
                    if row.get("mismatch") is not None
                ),
                None,
            )
        case = (
            None
            if failure is None
            else {
                key: failure.get(key)
                for key in ("prompt_id", "configuration_id", "repetition", "mismatch")
            }
        )
        lifecycle.output(
            "model_report",
            json.dumps(
                {
                    "state": status["state"],
                    "stage": status["stage"],
                    "error_type": status.get("error_type"),
                    "first_mismatch": case,
                    "checked_comparisons": report["checked_comparisons"],
                },
                sort_keys=True,
            ),
        )
        print(
            "TOKENOMICS_MODEL_SUMMARY " + json.dumps(report, sort_keys=True), flush=True
        )
    return 0 if code == 0 and status["state"] == "passed" else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--owned-child"]:
        owner = lifecycle.current()
        root = lifecycle.ownership(owner)
        sys.exit(child(root / "model-results"))
    if len(sys.argv) != 1:
        sys.exit("Fixed workload only")
    sys.exit(workload())
