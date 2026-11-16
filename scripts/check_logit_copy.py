"""CPU live-vs-canonical replay check for the frozen logit-copy validation."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.logit_copy import analyze, load_collection
from bench.common import ROOT, atomic_json, read_jsonl, sha256, source_fingerprint
from bench.model_identity import model_identity
from scripts.check_models import check_cycles, compare, execute

WORK = ("cycles", "drafted", "accepted", "verified_rows")


def check(args, *, expected_sha=None):
    targets, receipt = load_collection(args.collection)
    if receipt.get("collection_phase") != "validation":
        raise ValueError("Live check requires the frozen validation collection")
    synthetic = expected_sha is not None
    expected_sha = expected_sha or model_identity(receipt["model_id"])["sha256"]
    if (
        bool(receipt["synthetic_fixture"]) != synthetic
        or sha256(args.model) != expected_sha
        or receipt["model_sha256"] != expected_sha
    ):
        raise ValueError("Wrong live model or synthetic/learned scope")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = source_fingerprint()
    binary_sha = sha256(args.binary)
    report = {
        "source": source,
        "binary_sha256": binary_sha,
        "model_id": receipt["model_id"],
        "model_sha256": expected_sha,
        "collection_sha256": sha256(Path(args.collection) / "complete.json"),
        "synthetic_fixture": synthetic,
        "scope": "Predeclared six-prompt CPU development validation; exact IDs/stops and per-request replay work, no latency or held-out claim",
    }
    atomic_json(out / "manifest.json", report)
    checked = 0
    try:
        prediction = analyze(args.collection, out / "canonical-replay")
        expected = {
            (
                row["prompt_id"],
                row["configuration"]["id"],
                row["configuration"]["cap"],
            ): row
            for row in prediction["per_prompt"]
        }
        canonical = {row["prompt_id"]: row for row in targets}
        corpus = {
            row["id"]: row
            for row in read_jsonl(args.corpus or ROOT / "bench/prompts-v1.jsonl")
        }
        jobs = []
        configuration_keys = {}
        for target in targets:
            for config in (row for row in prediction["totals"]):
                ordinary = config["id"] == "ordinary"
                drafter = (
                    "logit-copy"
                    if config["family"] == "guided"
                    else (
                        "ngram"
                        if config["id"] == "upstream-ngram3"
                        else "suffix" if not ordinary else "model"
                    )
                )
                cid = config["id"] + f"-k{config['cap']}"
                configuration_keys[cid] = config["id"], config["cap"]
                jobs.append(
                    {
                        "prompt_id": target["prompt_id"],
                        "prompt": corpus[target["prompt_id"]]["prompt"],
                        "max_tokens": 64,
                        "configuration_id": cid,
                        "policy": "target" if ordinary else "fixed",
                        "k": config["cap"] if not ordinary else 4,
                        "drafter": drafter,
                        "repetition": 0,
                        "detailed": True,
                    }
                )
        settings = SimpleNamespace(
            binary=args.binary,
            target=args.model,
            out=out,
            context=2048,
            batch=512,
            threads=2,
            gpu_layers=0,
            timeout_seconds=1200,
        )
        audits = []

        def verify(row):
            nonlocal checked
            info = row["system_info"]
            if (
                "TOKENOMICS_CPU_BATCH_INVARIANT=ON" not in info
                or "TOKENOMICS_LOGIT_COPY=ON" not in info
                or any(
                    flag in info
                    for flag in (
                        "TOKENOMICS_GREEDY_NO_TEMP=ON",
                        "TOKENOMICS_GREEDY_DIRECT=ON",
                        "TOKENOMICS_CPU_Q4_PAIR=ON",
                        "TOKENOMICS_CPU_PACKED_VERIFY=ON",
                    )
                )
            ):
                raise ValueError(
                    "Live validation requires the original guarded CPU/sampler"
                )
            replay_id, replay_cap = configuration_keys[row["configuration_id"]]
            wanted = expected[(row["prompt_id"], replay_id, replay_cap)]
            mismatch = compare(canonical[row["prompt_id"]], row)
            work_mismatch = {
                key: [wanted[key], row["execution"][key]]
                for key in WORK
                if wanted[key] != row["execution"][key]
            }
            audit = {
                "prompt_id": row["prompt_id"],
                "configuration_id": row["configuration_id"],
                "token_mismatch": mismatch,
                "work_mismatch": work_mismatch,
            }
            audits.append(audit)
            atomic_json(out / "correctness.json", audits)
            checked += 1
            if mismatch is not None or work_mismatch:
                raise ValueError("Live output or work differs from canonical replay")

        rows = execute(settings, "live", jobs, False, verify)
        cycles = read_jsonl(out / "live.cycles.jsonl")
        check_cycles(rows, cycles)
        grouped = {}
        for cycle in cycles:
            grouped.setdefault(
                (cycle["prompt_id"], cycle["configuration_id"]), []
            ).append(cycle)
        for row in rows:
            if row["settings"]["drafter"] != "logit-copy":
                continue
            telemetry = row["logit_copy"]
            if any(
                telemetry.get(key) != value
                for key, value in (
                    ("minimum_match_length", 3),
                    ("maximum_match_length", 32),
                    ("hint_rank_cutoff", 64),
                    ("fallback_to_plain_copy", False),
                )
            ):
                raise ValueError("Wrong live logit-copy algorithm")
            sums = {
                key: 0
                for key in ("source_candidates", "rank_checks", "logits_examined")
            }
            cursor = 0
            for cycle in grouped[(row["prompt_id"], row["configuration_id"])]:
                trace = cycle["logit_copy"]
                if trace["hint_available"]:
                    if cursor <= 0 or trace["hint_output_index"] != cursor - 1:
                        raise ValueError(
                            "Live drafter consumed a future/wrong canonical row"
                        )
                elif trace["hint_output_index"] != -1:
                    raise ValueError("Unexpected hint without a distribution")
                for key in sums:
                    sums[key] += trace[key]
                cursor += cycle["committed"]
            if any(sums[key] != telemetry[key] for key in sums):
                raise ValueError("Guide telemetry does not sum to request totals")
        if (
            source_fingerprint() != source
            or sha256(args.binary) != binary_sha
            or sha256(args.model) != expected_sha
        ):
            raise ValueError("Live source/binary/model changed")
        totals = {}
        for row in rows:
            value = totals.setdefault(row["configuration_id"], {key: 0 for key in WORK})
            for key in WORK:
                value[key] += row["execution"][key]
        status = {
            "state": "passed",
            "checked_comparisons": checked,
            "same_work_as_canonical_replay": True,
            "causal_hint_audit": "passed",
            "synthetic_fixture": synthetic,
            "model_id": receipt["model_id"],
            "totals": totals,
            "performance_claim": None,
        }
        atomic_json(out / "status.json", status)
        names = (
            "manifest.json",
            "status.json",
            "correctness.json",
            "live.jobs.jsonl",
            "live.jsonl",
            "live.cycles.jsonl",
            "live.stderr.log",
        )
        hashes = {name: sha256(out / name) for name in names}
        atomic_json(out / "complete.json", {"sha256": hashes})
        print(
            "LIVE_LOGIT_COPY_SUMMARY="
            + json.dumps(
                {
                    **status,
                    "source": source,
                    "binary_sha256": binary_sha,
                    "model_sha256": expected_sha,
                    "payload_sha256": hashes,
                    "complete_sha256": sha256(out / "complete.json"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return status
    except Exception as error:
        atomic_json(
            out / "status.json",
            {
                "state": "failed",
                "checked_comparisons": checked,
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.corpus = None
    check(args)


if __name__ == "__main__":
    main()
