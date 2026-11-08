"""Audit a frozen paired CPU/logging grid without changing its raw artifact."""

import json
import tempfile
from pathlib import Path
from analysis.controller import verify as audit_grid
from analysis.profile_comparison import analyze
from bench.common import ROOT, sha256
from scripts.verify_controller_evidence import equal


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
