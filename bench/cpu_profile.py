"""Frozen CPU-profile identity and the trained prerequisite for a new study."""

import json
import re

from bench.common import sha256

STUDY_CONFIGS = {
    "guarded": "configs/controller-study-v1.json",
    "packed": "configs/controller-packed-study-v1.json",
}


def study_configuration(root, profile):
    if profile not in STUDY_CONFIGS:
        raise ValueError("Unsupported controller CPU profile")
    cfg = json.loads((root / STUDY_CONFIGS[profile]).read_text())
    if cfg.get("cpu_profile", "guarded") != profile:
        raise ValueError("Controller protocol CPU profile changed")
    return cfg


def profile_matches(build, profile):
    if profile not in STUDY_CONFIGS:
        raise ValueError("Unsupported controller CPU profile")
    values = build["values"]
    return values.get("TOKENOMICS_CPU_BATCH_INVARIANT") == (
        "ON" if profile == "guarded" else "OFF"
    ) and values.get("TOKENOMICS_CPU_PACKED_VERIFY") == (
        "ON" if profile == "packed" else "OFF"
    )


def verified_packed_gate(root, cfg):
    digest = cfg.get("trained_gate_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError(
            "A complete verified trained packed gate must be frozen first"
        )
    path = root / cfg["trained_gate"]
    if sha256(path) != digest:
        raise RuntimeError("Trained packed gate provenance changed")
    proof = json.loads(path.read_text())
    if (
        proof.get("state") != "kernel_profile_passed"
        or proof.get("exact_comparisons") != 272
        or proof.get("independent_output_checks") is not True
        or proof.get("shutdown_confirmed") is not True
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", proof.get(key, ""))
            for key in ("binary_sha256", "screen_target_sha256")
        )
    ):
        raise RuntimeError("A complete verified trained packed gate is required")
    return proof
