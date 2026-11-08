"""Recompute frozen controller priors/statistics from retained trained outputs."""

from __future__ import annotations
import json
import math
import tempfile
from pathlib import Path
from analysis.controller import report, verify
from bench.common import ROOT, read_jsonl, sha256
from bench.controller import calibrate
from scripts.check_models import compare, check_cycles

SOURCE = "e98c0fa84a2c2ff52df198f19eae6ce72d51c83c"
RUN = 42492089188


def equal(a, b):
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, list):
        return (
            isinstance(b, list)
            and len(a) == len(b)
            and all(equal(x, y) for x, y in zip(a, b))
        )
    if isinstance(a, float):
        return isinstance(b, (int, float)) and math.isclose(
            a, b, rel_tol=1e-12, abs_tol=1e-12
        )
    return type(a) is type(b) and a == b


def check(evidence, *, source=SOURCE, synthetic_fixture=False):
    """No inference. Synthetic acceptance is explicit and only used by tests."""
    cfg = json.loads((ROOT / "configs/controller-study-v1.json").read_text())
    status = json.loads((evidence / "status.json").read_text())
    if status["state"] != "controller_study_complete":
        raise ValueError("Controller study is incomplete")
    prior = calibrate(
        evidence / "fit",
        evidence / "replay",
        "ngram",
        synthetic_fixture=synthetic_fixture,
    )
    if prior != json.loads((evidence / "prior.json").read_text()):
        raise ValueError("Recomputed controller calibration differs")
    build = json.loads((evidence / "controller-build.json").read_text())
    if (
        build["binary_sha256"]
        != prior["provenance"]["operating_point"]["binary_sha256"]
        or build["configuration"]["values"]["TOKENOMICS_CPU_BATCH_INVARIANT"] != "ON"
        or build["configuration"]["values"]["TOKENOMICS_CPU_PACKED_VERIFY"] != "OFF"
    ):
        raise ValueError("Controller study changed its guarded execution profile")
    checked = {}
    for name in ("fit", "validation", "logging"):
        rows, manifest = verify(evidence / name)
        frozen = manifest["frozen"]
        if (
            frozen["code"]["revision"] != source
            or frozen["identity"]["synthetic_fixture"] != synthetic_fixture
        ):
            raise ValueError("Controller study source/fixture identity changed")
        checked[name] = len(rows)
    manifest = json.loads((evidence / "validation/run_manifest.json").read_text())[
        "frozen"
    ]
    actual = manifest["config"]
    if not synthetic_fixture and (
        any(actual.get(k) != v for k, v in cfg.items())
        or {p["id"] for p in manifest["prompts"]} != set(cfg["validation_prompt_ids"])
    ):
        raise ValueError("Controller study changed the predeclared validation grid")
    with tempfile.TemporaryDirectory() as tmp:
        derived = report(evidence / "validation", Path(tmp) / "validation", plots=False)
        logging = report(evidence / "logging", Path(tmp) / "logging", plots=False)
    for name, result in (("validation", derived), ("logging", logging)):
        archived = json.loads((evidence / name / "analysis/summary.json").read_text())
        if not equal(result, archived):
            raise ValueError("Controller summary differs from regenerated statistics")
    comparisons = 0
    observed = {}
    for drafter in ("model", "ngram"):
        directory = evidence / ("gate-" + drafter)
        gate = json.loads((directory / "status.json").read_text())
        receipt = json.loads((directory / "complete.json").read_text())
        for name in ("requests.jsonl", "cycles.jsonl"):
            if sha256(directory / name) != receipt["sha256"][name]:
                raise ValueError("Controller correctness artifact changed")
        rows, cycles = read_jsonl(directory / "requests.jsonl"), read_jsonl(
            directory / "cycles.jsonl"
        )
        if (
            gate["state"] != "passed"
            or not gate["correctness_only"]
            or not gate["synthetic_priors"]
            or gate["binary_sha256"] != build["binary_sha256"]
            or gate["records"] != len(rows)
            or receipt["jobs"] != len(rows)
        ):
            raise ValueError("Invalid trained controller correctness gate")
        refs = {r["repetition"]: r for r in rows if r["configuration_id"] == "target"}
        for row in rows:
            if compare(refs[row["repetition"]], row) is not None:
                raise ValueError("Trained controller output divergence")
            if row["settings"]["policy"] == "cost-aware" and (
                not row["controller_active"]
                or not row["settings"]["controller"]["synthetic"]
            ):
                raise ValueError("Controller correctness policy did not execute")
        check_cycles([r for r in rows if r["settings"]["detailed"]], cycles)
        observed[drafter] = gate["actions_observed"]
        comparisons += len(rows)
    if not synthetic_fixture and (
        checked["fit"] != 18 or checked["validation"] != 252 or comparisons != 56
    ):
        raise ValueError("Unexpected controller measurement/gate counts")
    return dict(
        exact_comparisons=checked["validation"] + comparisons,
        controller_gate_records=comparisons,
        controller_actions_observed=observed,
        verified_fit_records=checked["fit"],
        verified_validation_records=checked["validation"],
        verified_logging_records=checked["logging"],
        prior_id=prior["prior_id"],
        controller=derived,
        controller_logging=logging,
        peak_rss_bytes=status["peak_rss_bytes"],
        binary_sha256=build["binary_sha256"],
    )


def check_partial(evidence, *, source=SOURCE, synthetic_fixture=False):
    """Audit retained complete blocks without estimating an unfinished grid."""
    checked = {}
    for name in ("fit", "validation", "logging"):
        directory = evidence / name
        if not (directory / "run_manifest.json").exists():
            continue
        rows, manifest = verify(directory, complete_required=False)
        frozen = manifest["frozen"]
        if (
            frozen["code"]["revision"] != source
            or frozen["identity"]["synthetic_fixture"] != synthetic_fixture
        ):
            raise ValueError("Partial controller source/fixture identity changed")
        checked[name] = len(rows)
    if (evidence / "prior.json").exists():
        prior = calibrate(
            evidence / "fit",
            evidence / "replay",
            "ngram",
            synthetic_fixture=synthetic_fixture,
        )
        if prior != json.loads((evidence / "prior.json").read_text()):
            raise ValueError("Partial controller calibration differs")
    return dict(
        partial=True,
        exact_comparisons=sum(checked.values()),
        verified_records_by_stage=checked,
        limitation="Complete retained blocks only; no unfinished-grid performance estimate.",
    )
