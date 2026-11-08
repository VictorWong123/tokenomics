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


def check_diagnosis(evidence, expected_source):
    from bench.common import read_jsonl
    from bench.run import first_mismatch
    from scripts.check_models import compare, check_cycles, validate_record
    from scripts.diagnose_models import probe_at
    from types import SimpleNamespace

    result = json.loads((evidence / "diagnosis.json").read_text())
    spec = json.loads((ROOT / "configs/profile-diagnosis-v1.json").read_text())
    receipt_path = ROOT / spec["failure_receipt"]
    if sha256(receipt_path) != spec["failure_receipt_sha256"]:
        raise ValueError("Diagnostic failure binding changed")
    receipt = json.loads(receipt_path.read_text())
    source = receipt_path.parent / f"attempt-{receipt['attempt']}"
    original = check_failure(source, receipt["source_sha"])
    if (
        result["source"]["revision"] != expected_source
        or result["state"] != "profile_diagnosis_complete"
        or result["diagnostic_only"] is not True
        or result["performance_summary_produced"] is not False
        or result["both_original_histories_reproduced"] is not True
        or result["common_prefix_token_ids_equal"] is not True
    ):
        raise ValueError("Invalid diagnostic source/scope")
    frozen = json.loads((source / "profiles/run_manifest.json").read_text())["frozen"]
    block = next((source / "profiles/blocks").glob("*/attempt-*/failure.json")).parent
    rows = {}
    probes = {}
    for profile, cid in (("guarded", "target"), ("packed", "packed-target")):
        expected = next(
            r
            for r in read_jsonl(block / f"{cid}.requests.jsonl")
            if not r.get("warmup")
        )
        records = read_jsonl(evidence / profile / "original-history.jsonl")
        jobs = read_jsonl(evidence / profile / "original-history.jobs.jsonl")
        original_jobs = read_jsonl(block / f"{cid}.jobs.jsonl")
        original_rows = read_jsonl(block / f"{cid}.requests.jsonl")
        if len(records) != len(jobs) or len(jobs) != len(original_jobs):
            raise ValueError("Diagnostic warm-up/request count changed")
        for job, original_job, record, original_row in zip(
            jobs, original_jobs, records, original_rows
        ):
            allowed = dict(original_job)
            if allowed.get("warmup"):
                allowed["output_cap"] = allowed["max_tokens"]
            else:
                allowed.update(
                    detailed=True,
                    diagnostic_logits=True,
                    capture_checkpoints=[36],
                    checkpoint_namespace="profile-diagnosis",
                )
            if job != allowed or compare(original_row, record) is not None:
                raise ValueError("Diagnostic original jobs or warm-up history changed")
            validate_record(
                SimpleNamespace(**frozen["config"], gpu_layers=0), job, record
            )
        actual = [r for r in records if not r.get("warmup")]
        if len(actual) != 1 or compare(expected, actual[0]) is not None:
            raise ValueError("Diagnostic history does not reproduce retained IDs/stops")
        row = actual[0]
        if (
            not row["settings"]["diagnostic"]
            or not row["settings"]["diagnostic_logits"]
            or not row["settings"]["detailed"]
            or any(
                row["settings"][k] != expected["settings"][k]
                for k in ("context", "batch", "threads", "gpu_layers")
            )
            or result["binary_sha256"][profile]
            != frozen["configuration_binary_sha256"][cid]
        ):
            raise ValueError("Diagnostic execution identity changed")
        cycles = [
            c
            for c in read_jsonl(evidence / profile / "original-history.cycles.jsonl")
            if not c.get("warmup")
        ]
        check_cycles(actual, cycles)
        probes[profile] = [
            probe_at(cycles, cid, row["repetition"], pos, prompt_id=row["prompt_id"])
            for pos in (35, 36)
        ]
        if any(
            p["top_id"] != row["token_ids"][p["output_index"]] for p in probes[profile]
        ):
            raise ValueError("Diagnostic top ID differs from emitted greedy token")
        rows[profile] = row
    mismatch = first_mismatch(rows["guarded"]["token_ids"], rows["packed"]["token_ids"])
    if (
        mismatch != spec["expected_first_mismatch"]
        or mismatch != original["first_mismatch"]
        or mismatch != result["first_mismatch"]
        or not equal(probes, result["probes"])
        or result["target_sha256"] != frozen["target_sha256"]
        or rows["guarded"]["token_ids"][:36] != rows["packed"]["token_ids"][:36]
    ):
        raise ValueError("Diagnostic probes or mismatch differ from original evidence")
    return dict(
        exact_comparisons=2,
        exact_original_history_reproductions=2,
        diagnostic_only=True,
        performance_summary_produced=False,
        first_mismatch=mismatch,
        target_probes=probes,
        binary_sha256=result["binary_sha256"],
        screen_target_sha256=result["target_sha256"],
    )
