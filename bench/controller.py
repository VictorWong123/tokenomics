"""Freeze development-only priors and enforce their measured operating point."""

from __future__ import annotations
import copy
import hashlib
import json
import math
import pathlib
import re
import statistics
from analysis.report import cells, select_fixed, validate_run
from bench.common import read_jsonl, sha256

ACTIONS = (0, 1, 2, 4, 6, 8)
POINT_KEYS = (
    "binary_sha256",
    "target_sha256",
    "draft_sha256",
    "gpu_layers",
    "context",
    "batch",
    "threads",
    "drafter",
    "cache",
)


def _calibration_system_info(rows, configurations, drafter):
    drafted = {
        c["id"]
        for c in configurations
        if (c["policy"] != "target" and c.get("drafter", "model") == "model")
        or c.get("resident", False)
    }
    loaded = {r["system_info"] for r in rows if r["configuration_id"] in drafted}
    ordinary = {r["system_info"] for r in rows if r["configuration_id"] not in drafted}
    if drafter != "model":
        loaded |= ordinary
        ordinary.clear()
    if len(loaded) != 1 or ordinary and len(ordinary) != 1:
        raise ValueError("Mixed calibration backends")
    system_info = next(iter(loaded))
    if not ordinary or system_info == next(iter(ordinary)):
        return system_info
    target_info = next(iter(ordinary))
    offload = re.compile(r"; load_tensors: offloaded [^\n]*\n")
    target_loads = offload.findall(target_info)
    loaded_loads = offload.findall(system_info)
    if (
        loaded_loads[:-1] != target_loads
        or offload.sub("", system_info) != offload.sub("", target_info)
    ):
        raise ValueError("Mixed calibration backends")
    return system_info


def point_from_frozen(frozen, drafter, cache="retain", *, configuration_id=None):
    result = {
        **{k: frozen[k] for k in POINT_KEYS[:4]},
        **{k: frozen["config"][k] for k in ("context", "batch", "threads")},
        "drafter": drafter,
        "cache": cache,
    }
    if configuration_id is not None and "configuration_binary_sha256" in frozen:
        result["binary_sha256"] = frozen["configuration_binary_sha256"][
            configuration_id
        ]
    return result


def validate_binding(
    controller, expected, *, synthetic_fixture=False, system_info=None
):
    """Run before launch; additionally compare backend identity after generation."""
    if controller.get("synthetic") is not False and not (
        synthetic_fixture and controller.get("synthetic") is True
    ):
        raise ValueError(
            "Real-model controller performance requires measured, non-synthetic priors"
        )
    provenance = controller.get("provenance", {})
    point = provenance.get("operating_point", {})
    if any(point.get(k) != expected[k] for k in POINT_KEYS):
        raise ValueError(
            "Controller calibration operating point differs from executable/models/settings"
        )
    if (
        provenance.get("split") != "development"
        or not provenance.get("fit_prompt_ids")
        or len(set(provenance["fit_prompt_ids"])) != len(provenance["fit_prompt_ids"])
        or not provenance.get("sources")
        or not point.get("system_info")
    ):
        raise ValueError(
            "Controller needs frozen development provenance and backend identity"
        )
    if system_info is not None and point["system_info"] != system_info:
        raise ValueError("Controller runtime backend differs from calibration")


def freeze(
    fit_dir,
    replay_samples,
    replay_sources,
    drafter,
    *,
    synthetic_fixture=False,
    sync_samples=None,
):
    """Cycle priors come from original-state replays; fixed cap from full requests.

    replay_samples are independently checked by the capture/replay driver before
    entry. Never derive the fixed baseline from these hindsight one-cycle costs.
    """
    manifest = json.loads((fit_dir / "run_manifest.json").read_text())
    frozen = manifest["frozen"]
    rows = read_jsonl(fit_dir / "requests.jsonl")
    validate_run(rows, manifest, json.loads((fit_dir / "complete.json").read_text()))
    prompts = frozen["prompts"]
    if frozen["config"].get("split") != "development" or any(
        p.get("split") != "development" for p in prompts
    ):
        raise ValueError("Calibration can only use development fit prompts")
    ids = {p["id"] for p in prompts}
    if not replay_samples or any(s["prompt_id"] not in ids for s in replay_samples):
        raise ValueError("Replay priors must use the same development fit prompts")
    system_info = _calibration_system_info(
        rows, frozen["config"]["configurations"], drafter
    )
    priors = []
    for action in ACTIONS:
        samples = [s for s in replay_samples if s["action"] == action]
        if not samples:
            raise ValueError("Missing full-action calibration samples")
        for s in samples:
            if (
                s["effective_cap"] != action
                or s["committed"] < 1
                or s["committed"] > action + 1
                or s["synced"] < 0
                or not math.isfinite(s["observation_ms"])
                or not 0 <= s["sync_ms"] < s["observation_ms"]
            ):
                raise ValueError("Invalid or clipped calibration observation")
        costs = [
            s["observation_ms"] - (s["sync_ms"] if s["synced"] else 0) for s in samples
        ]
        if any(not math.isfinite(v) or v < 1e-9 or v > 1e12 for v in costs):
            raise ValueError("Invalid base cycle cost")
        priors.append(
            {
                "action": action,
                "committed_tokens": statistics.mean(s["committed"] for s in samples),
                "base_ms": statistics.mean(costs),
            }
        )
    positive_sync = [s for s in (sync_samples or []) if s["synced"] > 0]
    if drafter == "model" and not positive_sync:
        raise ValueError("Model calibration needs an actual catch-up episode")
    sync = (
        sum(s["sync_ms"] for s in positive_sync)
        / sum(s["synced"] for s in positive_sync)
        if positive_sync
        else 0
    )
    if any(
        not math.isfinite(s["sync_ms"]) or not 0 < s["sync_ms"] < s["observation_ms"]
        for s in positive_sync
    ):
        raise ValueError("Invalid synchronization calibration observation")
    selected = select_fixed(cells(rows))
    point = point_from_frozen(frozen, drafter)
    point["system_info"] = system_info
    sources = {
        "fit/run_manifest.json": sha256(fit_dir / "run_manifest.json"),
        "fit/requests.jsonl": sha256(fit_dir / "requests.jsonl"),
        "fit/complete.json": sha256(fit_dir / "complete.json"),
        **replay_sources,
    }
    result = {
        "schema_version": 1,
        "synthetic": bool(synthetic_fixture),
        "priors": priors,
        "alpha": 0.2,
        "switch_margin": 0.05,
        "sync_ms_per_token": sync,
        "min_hold_cycles": 4,
        "probe_interval": 16,
        "allow_target_only": True,
        "enable_probes": True,
        "provenance": {
            "split": "development",
            "fit_prompt_ids": sorted(ids),
            "operating_point": point,
            "sources": sources,
            "selected_fixed": selected,
            "selection_rule": "Full-request geometric mean of prompt/cap median decode throughput",
            "prior_rule": "Arithmetic mean of full-action original-state replay cycles",
            "samples_per_action": {
                str(a): sum(s["action"] == a for s in replay_samples) for a in ACTIONS
            },
        },
    }
    digest = hashlib.sha256(
        json.dumps(result, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    result["prior_id"] = "controller-v1-" + digest
    validate_binding(result, point, synthetic_fixture=synthetic_fixture)
    return result


def configurations(prior, *, detailed=False):
    fixed = prior["provenance"]["selected_fixed"]
    drafter = prior["provenance"]["operating_point"]["drafter"]
    result = [
        {"id": "target", "policy": "target"},
        {"id": fixed, "policy": "fixed", "k": int(fixed[5:]), "drafter": drafter},
        {
            "id": "dynamic",
            "policy": "confidence" if drafter == "model" else "fixed",
            "k": 8,
            "drafter": drafter,
        },
        {"id": "acceptance", "policy": "acceptance", "drafter": drafter},
    ]
    if drafter == "model":
        result.insert(1, {"id": "resident", "policy": "target", "resident": True})
    for name, override in (
        ("cost-aware", {}),
        ("no-fallback", {"allow_target_only": False}),
        ("no-probes", {"enable_probes": False}),
    ):
        result.append(
            {
                "id": name,
                "policy": "cost-aware",
                "drafter": drafter,
                "detailed": detailed,
                "controller": {**copy.deepcopy(prior), **override},
            }
        )
    return result


def calibrate(fit_dir, replay_dir, drafter, *, synthetic_fixture=False):
    """Recheck retained common-state replays, then freeze their six-action prior.

    Full request measurements select the fixed comparator; original-history
    action continuations supply priors. Transition episodes only fit sync cost.
    """
    fit_dir, replay_dir = pathlib.Path(fit_dir), pathlib.Path(replay_dir)
    from analysis.controller import verify

    verify(fit_dir)
    fit = json.loads((fit_dir / "run_manifest.json").read_text())["frozen"]
    manifest = json.loads((replay_dir / "manifest.json").read_text())
    expected = {
        "binary": fit["binary_sha256"],
        "target": fit["target_sha256"],
        "draft": fit["draft_sha256"],
        "config": fit["config"],
        "gpu_layers": fit["gpu_layers"],
        "drafter": drafter,
        "pilot": sha256(fit_dir / "run_manifest.json"),
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Replay calibration identity differs from development fit")
    if (
        bool(fit.get("identity", {}).get("synthetic_fixture", False))
        != synthetic_fixture
    ):
        raise ValueError("Synthetic calibration labeling differs from source run")
    refs = {
        r["prompt_id"]: r
        for r in read_jsonl(fit_dir / "requests.jsonl")
        if r["configuration_id"] == "target"
        and r["repetition"] == 0
        and r["output_cap"] == max(fit["config"]["output_caps"])
    }
    sources = {"replay/manifest.json": sha256(replay_dir / "manifest.json")}
    samples, syncs = [], []
    for done in sorted(replay_dir.glob("*/*/complete.json")):
        directory = done.parent
        receipt = json.loads(done.read_text())
        for name in ("requests.jsonl", "cycles.jsonl"):
            path = directory / name
            if sha256(path) != receipt.get("sha256", {}).get(name):
                raise ValueError("Replay calibration artifact hash changed")
            sources["replay/" + path.relative_to(replay_dir).as_posix()] = sha256(path)
        sources["replay/" + done.relative_to(replay_dir).as_posix()] = sha256(done)
        rows = read_jsonl(directory / "requests.jsonl")
        cycles = read_jsonl(directory / "cycles.jsonl")
        if len(rows) != receipt["jobs"]:
            raise ValueError("Incomplete calibration replay")
        for r in rows:
            pid, cid = r["prompt_id"], r["configuration_id"]
            if pid not in refs:
                raise ValueError("Unknown calibration fit prompt")
            ref = refs[pid]
            cs = [
                c
                for c in cycles
                if c["prompt_id"] == pid
                and c["configuration_id"] == cid
                and c["repetition"] == r["repetition"]
                and c.get("checkpoint") == r.get("checkpoint")
            ]
            if sum(c["committed"] for c in cs) != len(r["token_ids"]):
                raise ValueError("Calibration replay cycle accounting differs")
            if cid == "transition":
                if r["token_ids"] != ref["token_ids"][: len(r["token_ids"])]:
                    raise ValueError("Calibration transition output differs")
                syncs += [c for c in cs if c["synced"] > 0]
                continue
            if not cid.startswith("action") or not cid[6:].isdigit():
                continue
            action = int(cid[6:])
            if action not in ACTIONS:
                raise ValueError("Unsupported calibration action")
            cp = r["checkpoint"]
            if (
                len(cs) != 1
                or cs[0]["action"] != action
                or r["prompt_ids"] != ref["prompt_ids"] + ref["token_ids"][:cp]
                or r["settings"]["resume_checkpoint"] != f"{pid}:{cp}"
                or r["token_ids"] != ref["token_ids"][cp : cp + len(r["token_ids"])]
            ):
                raise ValueError(
                    "Calibration did not restore equivalent original state"
                )
            samples.append(cs[0])
    if {s["prompt_id"] for s in samples} != set(refs):
        raise ValueError("Every fit prompt needs common-state calibration samples")
    return freeze(
        fit_dir,
        samples,
        sources,
        drafter,
        synthetic_fixture=synthetic_fixture,
        sync_samples=syncs,
    )


if __name__ == "__main__":
    import argparse
    from bench.common import atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=pathlib.Path, required=True)
    parser.add_argument("--replay-dir", type=pathlib.Path, required=True)
    parser.add_argument("--drafter", choices=("model", "ngram"), required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--synthetic-fixture", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite frozen calibration")
    prior = calibrate(
        args.fit_dir,
        args.replay_dir,
        args.drafter,
        synthetic_fixture=args.synthetic_fixture,
    )
    atomic_json(args.out, prior)
    print(
        json.dumps(
            {
                "prior_id": prior["prior_id"],
                "synthetic": prior["synthetic"],
                "selected_fixed": prior["provenance"]["selected_fixed"],
            },
            sort_keys=True,
        )
    )
