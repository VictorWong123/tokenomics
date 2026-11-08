#!/usr/bin/env python3
"""Verify only the two frozen development runs; never start a Codespace."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import ROOT, atomic_json, read_jsonl, sha256
from scripts.check_models import check_cycles, compare

REPO = "VictorWong123/tokenomics"
EXPECTED = {
    42466570831: "af8b4044133ecef0de8751216248278a2fdb8ebd",
    42474834521: "63265d36939f802abb73b88962d96862c6c9169c",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def api(path, binary_out=None):
    args = ["gh", "api", f"repos/{REPO}/{path}"]
    if binary_out is not None:
        with binary_out.open("wb") as stream:
            subprocess.run(args, stdout=stream, check=True, timeout=120)
        return None
    result = subprocess.run(args, check=True, capture_output=True, timeout=120)
    return json.loads(result.stdout)


def unpack(archive, expected_zip_sha, out):
    """Fail closed before extracting any untrusted archive path."""
    require(sha256(archive) == expected_zip_sha, "Artifact ZIP digest mismatch")
    with zipfile.ZipFile(archive) as zipped:
        require(
            len(zipped.infolist()) == 2
            and set(zipped.namelist()) == {"report.json", "evidence.tar.gz"},
            "Unexpected ZIP members",
        )
        require(
            all(
                not z.is_dir() and z.file_size <= 64 * 1024**2
                for z in zipped.infolist()
            ),
            "Oversized ZIP member",
        )
        report = json.loads(zipped.read("report.json"))
        packed = out / "evidence.tar.gz"
        packed.write_bytes(zipped.read("evidence.tar.gz"))
    require(sha256(packed) == report["archive_sha256"], "Inner archive digest mismatch")
    evidence = out / "evidence"
    evidence.mkdir()
    with tarfile.open(packed) as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        require(
            len(names) == len(set(names)) and len(names) <= 100000,
            "Duplicate/excess archive members",
        )
        require(sum(m.size for m in members) <= 512 * 1024**2, "Oversized evidence")
        for member in members:
            path = PurePosixPath(member.name)
            require(
                member.isfile()
                and not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in member.name
                and member.size <= 32 * 1024**2,
                "Unsafe evidence member",
            )
        manifest = json.load(tar.extractfile("execution-manifest.json"))
        require(
            set(manifest["files"]) == set(names) - {"execution-manifest.json"},
            "Manifest member set mismatch",
        )
        for name, meta in manifest["files"].items():
            member = tar.getmember(name)
            require(member.size == meta["bytes"], "Member size mismatch")
            digest = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            require(digest.hexdigest() == meta["sha256"], "Member digest mismatch")
        tar.extractall(evidence, filter="data")
    for key in ("run_id", "attempt", "sha"):
        require(
            str(report[key]) == str(manifest[key]), "Archive/report ownership mismatch"
        )
    return report, manifest, evidence


def verify_grid(directory, complete_required, expected_source):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    frozen = manifest["frozen"]
    cfg = frozen["config"]
    require(frozen["code"]["revision"] == expected_source, "Measurement source changed")
    rows = read_jsonl(directory / "requests.jsonl")
    checks = read_jsonl(directory / "correctness.jsonl")
    require(
        len(checks) == len(rows) and all(c["ok"] for c in checks),
        "Non-passing correctness evidence",
    )
    expected = {
        (p["id"], cap, c["id"], rep)
        for p in frozen["prompts"]
        for cap in cfg["output_caps"]
        for c in cfg["configurations"]
        for rep in range(cfg["repetitions"])
    }
    keys = [
        (r["prompt_id"], r["output_cap"], r["configuration_id"], r["repetition"])
        for r in rows
    ]
    require(
        len(keys) == len(set(keys)) and set(keys) <= expected,
        "Unexpected/duplicate grid record",
    )
    require(not complete_required or set(keys) == expected, "Incomplete required grid")
    configurations = {c["id"]: c for c in cfg["configurations"]}
    groups = {}
    for row in rows:
        conf = configurations[row["configuration_id"]]
        for key in ("policy", "k", "cache", "drafter", "confidence", "detailed"):
            if key in conf:
                require(row["settings"][key] == conf[key], "Condition setting changed")
        require(
            row["settings"]["max_tokens"] == row["output_cap"], "Output budget changed"
        )
        require(
            row["correctness"] == "pass"
            and row["generated_tokens"] == len(row["token_ids"]),
            "Invalid request count/status",
        )
        require(
            row["peak_rss_bytes"] > 0 and row["generation_ms"] > 0,
            "Invalid resource/timing measurement",
        )
        for key in ("context", "batch", "threads"):
            require(row["settings"][key] == cfg[key], "Execution setting changed")
        require(
            row["settings"]["gpu_layers"] == frozen["gpu_layers"], "Offload changed"
        )
        cycles = [
            c
            for c in read_jsonl(directory / row["source_cycles"])
            if not c.get("warmup")
        ]
        check_cycles([row], cycles)
        groups.setdefault(
            (row["prompt_id"], row["output_cap"], row["repetition"]), []
        ).append(row)
    for group in groups.values():
        require(
            {r["configuration_id"] for r in group}
            == {c["id"] for c in cfg["configurations"]},
            "Partial comparison block",
        )
        reference = next(r for r in group if r["configuration_id"] == "target")
        require(
            all(compare(reference, r) is None for r in group), "Greedy output mismatch"
        )
    markers = list((directory / "blocks").glob("*/complete.json"))
    require(len(markers) == len(groups), "Incomplete block markers")
    for marker in markers:
        data = json.loads(marker.read_text())
        for name, digest in data["sha256"].items():
            require(
                sha256(marker.parent / name) == digest,
                "Completed block digest mismatch",
            )
    return rows, frozen, len(groups)


def verify_kernel(evidence):
    gate = evidence / "kernel-gate"
    status = json.loads((gate / "status.json").read_text())
    require(
        status["state"] == "passed" and status["synthetic_fixture"] is False,
        "Trained gate not passed",
    )
    refs = read_jsonl(gate / "reference.jsonl") + read_jsonl(
        gate / "boundary-reference.jsonl"
    )
    reference = {r["prompt_id"]: r for r in refs}
    candidates = read_jsonl(gate / "candidate.jsonl")
    checks = read_jsonl(gate / "correctness.jsonl")
    require(
        len(candidates) == len(checks) == 272 and all(c["ok"] for c in checks),
        "Incomplete 272-comparison gate",
    )
    require(
        all(compare(reference[r["prompt_id"]], r) is None for r in candidates),
        "Trained greedy mismatch",
    )
    require(all(r["peak_rss_bytes"] > 0 for r in candidates), "Missing measured RSS")
    check_cycles(candidates, read_jsonl(gate / "candidate.cycles.jsonl"))
    diagnosis = evidence / "kernel-diagnostic"
    diagnostic = json.loads((diagnosis / "status.json").read_text())
    full = read_jsonl(diagnosis / "full.jsonl")
    drefs = {r["repetition"]: r for r in read_jsonl(diagnosis / "reference.jsonl")}
    require(
        diagnostic["reference_reset_equal"]
        and len(full) == len(diagnostic["full_comparisons"]) == 14
        and all(c["mismatch"] is None for c in diagnostic["full_comparisons"])
        and all(compare(drefs[r["repetition"]], r) is None for r in full),
        "Full-generation diagnosis mismatch",
    )
    point = evidence / "qwen25-05-3-q4-ngram/timing"
    rows, frozen, _ = verify_grid(point, True, EXPECTED[42474834521])
    require(
        len(rows) == 18 and frozen["identity"]["synthetic_fixture"] is False,
        "Incomplete trained screen",
    )
    marker = "TOKENOMICS_CPU_PACKED_VERIFY=ON"
    require(
        all(marker in r["system_info"] for r in candidates + full + rows),
        "Wrong CPU profile",
    )
    build = json.loads((evidence / "kernel-build.json").read_text())
    require(build["binary_sha256"] == frozen["binary_sha256"], "Screen binary changed")
    gate_manifest = json.loads((gate / "manifest.json").read_text())
    require(
        gate_manifest["binary_sha256"] == frozen["binary_sha256"],
        "Trained gate binary changed",
    )
    fit_cfg = json.loads((ROOT / "configs/ngram-confirmation-v1.json").read_text())
    fit = json.loads((ROOT / fit_cfg["fit_manifest"]).read_text())["frozen"]
    require(frozen["target_sha256"] == fit["target_sha256"], "Screen target changed")
    replay = evidence / "checkpoint-validation"
    replay_manifest = json.loads((replay / "manifest.json").read_text())
    require(
        replay_manifest["binary"] == frozen["binary_sha256"]
        and replay_manifest["target"] == frozen["target_sha256"]
        and replay_manifest["draft"] is None,
        "Checkpoint operating point changed",
    )
    canonical = {r["prompt_id"]: r for r in rows if r["configuration_id"] == "target"}
    actions = captures = 0
    for done in replay.glob("*/checkpoint-*/complete.json"):
        directory = done.parent
        complete = json.loads(done.read_text())
        for name, digest in complete["sha256"].items():
            require(sha256(directory / name) == digest, "Replay digest mismatch")
        rr = read_jsonl(directory / "requests.jsonl")
        check_cycles(rr, read_jsonl(directory / "cycles.jsonl"))
        for row in rr:
            ref = canonical[row["prompt_id"]]
            cp = row["checkpoint"]
            if row["configuration_id"] == "capture":
                require(
                    row["prompt_ids"] == ref["prompt_ids"]
                    and row["token_ids"] == ref["token_ids"][:cp],
                    "Capture differs from original history",
                )
                require(
                    row["checkpoint_ids"]
                    == [f"{row['prompt_id']}:{cp-8}", f"{row['prompt_id']}:{cp}"],
                    "Wrong saved checkpoints",
                )
                captures += 1
            else:
                cp -= 8 if row["configuration_id"] == "signal" else 0
                require(
                    row["prompt_ids"] == ref["prompt_ids"] + ref["token_ids"][:cp]
                    and row["token_ids"]
                    == ref["token_ids"][cp : cp + len(row["token_ids"])]
                    and row["settings"]["resume_checkpoint"]
                    == f"{row['prompt_id']}:{cp}",
                    "Action did not restore/continue the original checkpoint",
                )
                actions += 1
    require(captures == 3 and actions == 66, "Incomplete trained checkpoint path")
    return {
        "exact_comparisons": 272,
        "diagnostic_comparisons": 14,
        "timing_records": 18,
        "checkpoint_captures": captures,
        "checkpoint_actions": actions,
        "binary_sha256": frozen["binary_sha256"],
        "screen_target_sha256": frozen["target_sha256"],
        "peak_rss_bytes": max(r["peak_rss_bytes"] for r in candidates + rows),
        "screen": json.loads((point.parent / "analysis/measurements.json").read_text()),
    }


def verify_confirmation(evidence, state):
    rows, frozen, blocks = verify_grid(
        evidence / "confirmation",
        state == "confirmation_complete",
        EXPECTED[42466570831],
    )
    cfg = json.loads((ROOT / "configs/ngram-confirmation-v1.json").read_text())
    fit = json.loads((ROOT / cfg["fit_manifest"]).read_text())["frozen"]
    require(
        frozen["config"] == cfg and frozen["identity"]["synthetic_fixture"] is False,
        "Confirmation configuration changed",
    )
    require(
        frozen["binary_sha256"] == fit["binary_sha256"]
        and frozen["target_sha256"] == fit["target_sha256"],
        "Frozen fit identity changed",
    )
    result = {"exact_comparisons": len(rows), "verified_blocks": blocks}
    if state == "confirmation_complete":
        from analysis.report import summarize

        derived = summarize(rows, frozen_fixed=cfg["selected_fixed"])
        archived = json.loads(
            (evidence / "confirmation/analysis/summary.json").read_text()
        )
        require(
            len(rows) == 162
            and blocks == 54
            and archived["full_feasibility_cleared"] is False,
            "Incorrect completion claim",
        )
        for name, metric in derived["comparisons"].items():
            other = archived["comparisons"][name]
            require(
                math.isclose(metric["ratio"], other["ratio"], rel_tol=1e-12),
                "Derived ratio mismatch",
            )
            require(
                len(other["ci95"]) == 2
                and all(
                    math.isclose(a, b, rel_tol=1e-12)
                    for a, b in zip(metric["ci95"], other["ci95"])
                ),
                "Derived interval mismatch",
            )
        result["confirmation"] = archived
    return result


def retain(evidence, receipt_path):
    """Retain verified bytes per attempt; preserve the first canonical receipt."""
    receipt = json.loads(receipt_path.read_text())
    run_id = receipt["run_id"]
    require(run_id in EXPECTED, "Unknown retained run")
    folder = (
        "codespace-confirmation-20261108"
        if run_id == 42466570831
        else "codespace-packed-20261108"
    )
    root = ROOT / "results/verification" / folder
    destination = root / f"attempt-{receipt['attempt']}"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(evidence.rglob("*")):
        require(not source.is_symlink(), "Unsafe retained symlink")
        if source.is_file():
            target = destination / source.relative_to(evidence)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require(
                    sha256(target) == sha256(source), "Retained raw bytes are immutable"
                )
            else:
                shutil.copy2(source, target)
    canonical = root / "verification.json"
    if canonical.exists():
        previous = json.loads(canonical.read_text())
        for key in (
            "run_id",
            "attempt",
            "source_sha",
            "state",
            "zip_sha256",
            "archive_sha256",
            "exact_comparisons",
            "shutdown_confirmed",
        ):
            require(previous[key] == receipt[key], "Canonical point changed")
        require(
            previous["independent_output_checks"] is True,
            "Unverified canonical receipt",
        )
    else:
        shutil.copy2(receipt_path, canonical)
    receipt_path.write_bytes(canonical.read_bytes())


def verify_run(run_id, out, publish=False):
    run = api(f"actions/runs/{run_id}")
    require(run["head_sha"] == EXPECTED[run_id], "Unexpected source SHA")
    if run["status"] != "completed":
        print(f"TOKENOMICS_VERIFICATION_WAITING {run_id}", flush=True)
        return
    attempt = run["run_attempt"]
    name = f"tokenomics-development-timing-{run_id}-{attempt}"
    artifacts = api(f"actions/runs/{run_id}/artifacts?per_page=100")["artifacts"]
    matches = [a for a in artifacts if a["name"] == name and not a["expired"]]
    require(len(matches) == 1, "Missing/ambiguous current-attempt artifact")
    artifact = matches[0]
    require(artifact["size_in_bytes"] <= 64 * 1024**2, "Oversized artifact")
    digest = artifact["digest"]
    require(digest.startswith("sha256:"), "Missing GitHub ZIP digest")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        archive = root / "artifact.zip"
        api(f"actions/artifacts/{artifact['id']}/zip", archive)
        report, manifest, evidence = unpack(
            archive, digest.removeprefix("sha256:"), root
        )
        require(
            manifest["sha"] == EXPECTED[run_id]
            and str(manifest["run_id"]) == str(run_id)
            and str(manifest["attempt"]) == str(attempt),
            "Wrong run ownership",
        )
        state = report["state"]
        if run_id == 42466570831:
            require(state in ("paused", "confirmation_complete"), "Confirmation failed")
            checked = verify_confirmation(evidence, state)
        else:
            require(state == "kernel_profile_passed", "Experimental CPU profile failed")
            checked = verify_kernel(evidence)
        jobs = api(f"actions/runs/{run_id}/jobs?per_page=100")["jobs"]
        finalizers = [
            j for j in jobs if j["name"] == "finalize" and j["conclusion"] == "success"
        ]
        require(len(finalizers) == 1, "Missing successful finalizer")
        log_result = subprocess.run(
            ["gh", "api", f"repos/{REPO}/actions/jobs/{finalizers[0]['id']}/logs"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        shutdown_origin = "direct GitHub job-log API"
        if log_result.returncode == 0:
            markers = [
                json.loads(m.group(1))
                for line in log_result.stdout.splitlines()
                if (m := re.search(r"TOKENOMICS_CONTROLLER_REPORT (\{.*\})", line))
            ]
        else:
            # The connected GitHub reader can retain the exact controller report
            # when a hosted job's read token cannot download another job's logs.
            # Never infer Shutdown merely from a successful inference workload.
            folder = (
                "codespace-confirmation-20261108"
                if run_id == 42466570831
                else "codespace-packed-20261108"
            )
            retained = ROOT / "results/verification" / folder / "lifecycle.json"
            require(
                retained.is_file(),
                "Lifecycle log unavailable; retained API report required",
            )
            lifecycle = json.loads(retained.read_text())
            require(
                lifecycle["source_job_id"] == finalizers[0]["id"],
                "Wrong retained finalizer",
            )
            markers = [lifecycle["controller_report"]]
            shutdown_origin = str(retained.relative_to(ROOT))
            print("TOKENOMICS_RETAINED_LIFECYCLE " + shutdown_origin, flush=True)
        require(
            len(markers) == 1
            and markers[0]["passed"] is True
            and markers[0]["shutdown_confirmed"] is True
            and markers[0]["sha"] == EXPECTED[run_id]
            and str(markers[0]["run_id"]) == str(run_id)
            and str(markers[0]["attempt"]) == str(attempt),
            "Actual Shutdown/ownership not confirmed",
        )
        receipt = {
            "schema_version": 1,
            "state": state,
            "run_id": run_id,
            "attempt": attempt,
            "source_sha": EXPECTED[run_id],
            "artifact_id": artifact["id"],
            "zip_sha256": digest.removeprefix("sha256:"),
            "archive_sha256": report["archive_sha256"],
            "verified_files": len(manifest["files"]),
            "shutdown_confirmed": True,
            "shutdown_evidence_origin": shutdown_origin,
            "shutdown_source_job_id": finalizers[0]["id"],
            "verifier_sha256": sha256(Path(__file__)),
            "independent_output_checks": True,
            "full_feasibility_cleared": False,
            **checked,
        }
        path = out / f"{run_id}.json"
        atomic_json(path, receipt)
        if publish:
            retain(evidence, path)
        print(
            "TOKENOMICS_VERIFIED_DOCUMENT "
            + json.dumps(
                {
                    "run_id": run_id,
                    "sha256": sha256(path),
                    "content": path.read_text(),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Copy verified bytes under results/verification; never start inference",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for run_id in EXPECTED:
        verify_run(run_id, args.out, args.publish)
