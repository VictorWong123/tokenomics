"""Audit a frozen paired CPU/logging grid without changing its raw artifact."""

import json
import tempfile
from pathlib import Path
from analysis.controller import verify as audit_grid
from analysis.profile_comparison import analyze
from bench.common import ROOT, sha256
from scripts.verify_controller_evidence import equal


def check_failure(evidence, expected_source):
    """Audit the preserved cross-profile failure; produce no timing estimate."""
    from bench.common import read_jsonl
    from bench.run import first_mismatch
    from scripts.check_models import compare, check_cycles, validate_record
    from types import SimpleNamespace

    directory = evidence / "profiles"
    frozen = json.loads((directory / "run_manifest.json").read_text())["frozen"]
    cfg = json.loads((ROOT / "configs/cpu-profile-screen-v1.json").read_text())
    if frozen["code"]["revision"] != expected_source or frozen["config"] != cfg:
        raise ValueError("Failed profile source/grid changed")
    if frozen["identity"]["synthetic_fixture"] is not False:
        raise ValueError("Expected trained failure evidence")
    proof_path = ROOT / cfg["trained_gate"]
    if sha256(proof_path) != cfg["trained_gate_sha256"]:
        raise ValueError("Failed profile prerequisite changed")
    proof = json.loads(proof_path.read_text())
    if (
        frozen["configuration_binary_sha256"]["packed-target"] != proof["binary_sha256"]
        or frozen["target_sha256"] != proof["screen_target_sha256"]
    ):
        raise ValueError("Failed profile operating point changed")
    failures = list(directory.glob("blocks/*/attempt-*/failure.json"))
    if len(failures) != 1 or list(directory.glob("blocks/*/complete.json")):
        raise ValueError("Expected one preserved first-block failure")
    attempt = failures[0].parent
    checks = json.loads((attempt / "correctness.json").read_text())
    rows = {}
    for conf in cfg["configurations"]:
        cid = conf["id"]
        jobs = read_jsonl(attempt / f"{cid}.jobs.jsonl")
        records = read_jsonl(attempt / f"{cid}.requests.jsonl")
        if len(jobs) != len(records):
            raise ValueError("Incomplete failed-profile process")
        for job, row in zip(jobs, records):
            if row["settings"]["detailed"] != conf["detailed"] or row.get(
                "warmup", False
            ) != job.get("warmup", False):
                raise ValueError("Failed-profile logging/warm-up identity changed")
            # The benchmark warm-up retains the measured cap as metadata while
            # its actual max_tokens remains the separate excluded warm-up cap.
            checked_row = (
                dict(row, output_cap=row["settings"]["max_tokens"])
                if row.get("warmup")
                else row
            )
            validate_record(SimpleNamespace(**cfg, gpu_layers=0), job, checked_row)
        measured = [r for r in records if not r["warmup"]]
        if len(measured) != 1:
            raise ValueError("Ambiguous failed-profile request")
        row = measured[0]
        marker = (
            "TOKENOMICS_CPU_PACKED_VERIFY=ON"
            if conf["kernel_profile"] == "packed"
            else "TOKENOMICS_CPU_BATCH_INVARIANT=ON"
        )
        if marker not in row["system_info"]:
            raise ValueError("Failed-profile execution marker changed")
        if row["settings"]["detailed"]:
            check_cycles(
                measured,
                [
                    c
                    for c in read_jsonl(attempt / f"{cid}.cycles.jsonl")
                    if not c.get("warmup")
                ],
            )
        elif read_jsonl(attempt / f"{cid}.cycles.jsonl"):
            raise ValueError("Minimal failed request retained cycles")
        rows[cid] = row
    for profile, ref in (("guarded", "target"), ("packed", "packed-target")):
        for conf in cfg["configurations"]:
            if (
                conf["kernel_profile"] == profile
                and compare(rows[ref], rows[conf["id"]]) is not None
            ):
                raise ValueError(
                    "Failure is within a profile, not the recorded cross-profile discrepancy"
                )
    derived = {
        cid: first_mismatch(rows["target"]["token_ids"], r["token_ids"])
        for cid, r in rows.items()
    }
    if not derived["packed-target"] or {c["configuration_id"] for c in checks} != set(
        rows
    ):
        raise ValueError("Expected cross-profile mismatch was not observed")
    for c in checks:
        mismatch = derived[c["configuration_id"]]
        if (
            c["first_mismatch"] != mismatch
            or c["ok"] != (mismatch is None)
            or c["prompt_mismatch"] is not None
        ):
            raise ValueError(
                "Saved failure differs from independently compared outputs"
            )
    return dict(
        exact_comparisons=0,
        observed_requests=len(rows),
        within_profile_nonself_checks=4,
        correctness_passed=False,
        first_mismatch=derived["packed-target"],
        cross_profile_mismatches=derived,
        performance_summary_produced=False,
        binary_sha256=proof["binary_sha256"],
        reference_binary_sha256=frozen["binary_sha256"],
        screen_target_sha256=frozen["target_sha256"],
    )


def check(evidence, expected_source, *, synthetic_fixture=False):
    directory = evidence / "profiles"
    rows, manifest = audit_grid(directory)
    frozen = manifest["frozen"]
    cfg = json.loads((ROOT / "configs/cpu-profile-screen-v1.json").read_text())
    if (
        frozen["code"]["revision"] != expected_source
        or frozen["config"] != cfg
        or frozen["identity"]["synthetic_fixture"] != synthetic_fixture
    ):
        raise ValueError("Paired profile source/grid identity changed")
    proof_path = ROOT / cfg["trained_gate"]
    if sha256(proof_path) != cfg["trained_gate_sha256"]:
        raise ValueError("Paired profile trained proof changed")
    proof = json.loads(proof_path.read_text())
    if (
        proof["state"] != "kernel_profile_passed"
        or proof["exact_comparisons"] != 272
        or not proof["independent_output_checks"]
        or not proof["shutdown_confirmed"]
        or frozen["configuration_binary_sha256"]["packed-target"]
        != proof["binary_sha256"]
        or frozen["target_sha256"] != proof["screen_target_sha256"]
    ):
        raise ValueError("Paired profile differs from verified trained operating point")
    with tempfile.TemporaryDirectory() as tmp:
        derived = analyze(directory, Path(tmp), plots=False)
    archived = json.loads((directory / "analysis/summary.json").read_text())
    if not equal(derived, archived):
        raise ValueError("Paired profile statistics differ from audited records")
    return dict(
        exact_comparisons=len(rows),
        verified_blocks=len({r["block_id"] for r in rows}),
        paired_profile=derived,
        binary_sha256=proof["binary_sha256"],
        reference_binary_sha256=frozen["binary_sha256"],
        screen_target_sha256=frozen["target_sha256"],
        peak_rss_bytes=max(r["peak_rss_bytes"] for r in rows),
    )
