#!/usr/bin/env python3
"""Reproduce a retained CPU-profile divergence; diagnostic only, never throughput."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import ROOT, atomic_json, read_jsonl, sha256, source_fingerprint
from bench.cpu_profile import verified_packed_gate, profile_matches
from bench.run import first_mismatch
from scripts.check_kernel_profile import command
from scripts.check_models import execute, compare, check_cycles
from scripts.diagnose_models import build_configuration, probe_at
from scripts.download_models import download
from scripts.verify_profile_evidence import check_failure


def diagnose(executables, target, cfg, source, out):
    block = next((source / "profiles/blocks").glob("*/attempt-*/failure.json")).parent
    reference = {}
    probes = {}
    for profile, cid in (("guarded", "target"), ("packed", "packed-target")):
        expected = next(
            r
            for r in read_jsonl(block / f"{cid}.requests.jsonl")
            if not r.get("warmup")
        )
        jobs = copy.deepcopy(read_jsonl(block / f"{cid}.jobs.jsonl"))
        for j in jobs:
            if j.get("warmup"):
                j["output_cap"] = j["max_tokens"]  # metadata normalization only
            else:
                j.update(
                    detailed=True,
                    diagnostic_logits=True,
                    capture_checkpoints=[36],
                    checkpoint_namespace="profile-diagnosis",
                )
        directory = out / profile
        directory.mkdir(parents=True, exist_ok=True)
        a = SimpleNamespace(
            binary=executables[profile],
            target=target,
            draft=None,
            out=directory,
            context=cfg["context"],
            batch=cfg["batch"],
            threads=cfg["threads"],
            gpu_layers=0,
            timeout_seconds=600,
        )
        rows = execute(a, "original-history", jobs, False)
        row = next(r for r in rows if not r.get("warmup"))
        if compare(expected, row) is not None:
            raise RuntimeError(
                "Diagnostic request does not reproduce the retained output IDs/stops"
            )
        cycles = [
            c
            for c in read_jsonl(directory / "original-history.cycles.jsonl")
            if not c.get("warmup")
        ]
        check_cycles([row], cycles)
        if (
            not row["settings"]["diagnostic"]
            or not row["settings"]["diagnostic_logits"]
        ):
            raise RuntimeError("Diagnostic logits were not executed")
        reference[profile] = row
        probes[profile] = [
            probe_at(cycles, cid, row["repetition"], pos, prompt_id=row["prompt_id"])
            for pos in (35, 36)
        ]
        if any(
            p["top_id"] != row["token_ids"][p["output_index"]] for p in probes[profile]
        ):
            raise RuntimeError(
                "Diagnostic top ID does not match upstream greedy output"
            )
    mismatch = first_mismatch(
        reference["guarded"]["token_ids"], reference["packed"]["token_ids"]
    )
    result = dict(
        schema_version=1,
        state="profile_diagnosis_complete",
        diagnostic_only=True,
        first_mismatch=mismatch,
        common_prefix_token_ids_equal=(
            reference["guarded"]["token_ids"][:36]
            == reference["packed"]["token_ids"][:36]
        ),
        both_original_histories_reproduced=True,
        probes=probes,
        binary_sha256={p: sha256(x) for p, x in executables.items()},
        target_sha256=sha256(target),
        source=source_fingerprint(),
        performance_summary_produced=False,
        limitation="Different CPU arithmetic/configuration can change greedy choices. Original incremental histories and excluded warm-up are reproduced. This does not isolate the responsible operator or establish a speedup.",
    )
    atomic_json(out / "diagnosis.json", result)
    return result


def pipeline(out, model_dir):
    spec = json.loads((ROOT / "configs/profile-diagnosis-v1.json").read_text())
    receipt_path = ROOT / spec["failure_receipt"]
    if sha256(receipt_path) != spec["failure_receipt_sha256"]:
        raise RuntimeError("Frozen failure receipt changed")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt["state"] != "failed"
        or receipt["performance_summary_produced"]
        or not receipt["shutdown_confirmed"]
    ):
        raise RuntimeError(
            "A retained checked failure with actual Shutdown is required"
        )
    source = receipt_path.parent / f"attempt-{receipt['attempt']}"
    manifest = json.loads((source / "execution-manifest.json").read_text())
    for name, meta in manifest["files"].items():
        p = source / name
        if (
            p.is_symlink()
            or not p.resolve().is_relative_to(source.resolve())
            or sha256(p) != meta["sha256"]
            or p.stat().st_size != meta["bytes"]
        ):
            raise RuntimeError("Retained failure payload changed")
    failure = check_failure(source, receipt["source_sha"])
    if failure["first_mismatch"] != spec["expected_first_mismatch"]:
        raise RuntimeError("Recorded failure differs from diagnostic protocol")
    frozen = json.loads((source / "profiles/run_manifest.json").read_text())["frozen"]
    cfg = frozen["config"]
    proof = verified_packed_gate(ROOT, cfg)
    build = ROOT / "build-cpu"
    guard_dir = ROOT / "build-diagnosis-guarded"
    guard_dir.mkdir(exist_ok=False)
    for n in ("tokenomics", "CMakeCache.txt"):
        shutil.copy2(build / n, guard_dir / n)
    guarded = guard_dir / "tokenomics"
    if (
        not profile_matches(build_configuration(guarded), "guarded")
        or sha256(guarded) != frozen["binary_sha256"]
    ):
        raise RuntimeError("Guarded diagnostic executable differs from failure")
    py = Path(sys.executable)
    command(
        [
            py.with_name("cmake"),
            "-S",
            ROOT,
            "-B",
            build,
            "-DTOKENOMICS_CPU_BATCH_INVARIANT=OFF",
            "-DTOKENOMICS_CPU_PACKED_VERIFY=ON",
        ],
        120,
    )
    command([py.with_name("cmake"), "--build", build, "-j2"], 1200)
    packed = build / "tokenomics"
    if sha256(packed) != proof["binary_sha256"]:
        raise RuntimeError("Packed diagnostic executable differs from trained gate")
    target = download("qwen25-3", model_dir, model_dir / "models.lock.json")
    if sha256(target) != frozen["target_sha256"]:
        raise RuntimeError("Diagnostic target differs from failed comparison")
    result = diagnose({"guarded": guarded, "packed": packed}, target, cfg, source, out)
    if result["first_mismatch"] != spec["expected_first_mismatch"]:
        raise RuntimeError("Diagnostic did not reproduce the original divergence")
    atomic_json(
        out / "status.json",
        dict(
            state="profile_diagnosis_complete",
            diagnostic_only=True,
            performance_summary_produced=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    a = p.parse_args()
    a.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        pipeline(a.run_dir, a.model_dir)
    except Exception as exc:
        atomic_json(
            a.run_dir / "status.json",
            dict(state="failed", error_type=type(exc).__name__, message=str(exc)),
        )
        raise
