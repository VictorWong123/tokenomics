"""Bounded canonical CPU rank collection for a predeclared exploratory fit."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import (
    ROOT,
    atomic_json,
    read_jsonl,
    sha256,
    source_fingerprint,
    write_jsonl,
)
from bench.model_identity import DEFAULT_MODEL, MODEL_PINS, model_identity

PROTOCOL = ROOT / "configs/logit-copy-fit-v1.json"
PROTOCOL_SHA256 = "028018e39653338e2454c6d3a7e61ad01fabd314afed6cf5ee4d972b637f1cec"


VALIDATION_PROTOCOL = ROOT / "configs/logit-copy-validation-v1.json"
VALIDATION_PROTOCOL_SHA256 = (
    "99148b182a20c8e8805252542c55638e181d4e236646aed3cd69849c6a7d9124"
)


def collection_protocol(phase):
    if phase == "fit":
        return PROTOCOL, PROTOCOL_SHA256
    if phase == "validation":
        return VALIDATION_PROTOCOL, VALIDATION_PROTOCOL_SHA256
    raise ValueError("Unsupported canonical collection phase")


def collect(args, *, expected_sha=None, corpus_path=None):
    phase = getattr(args, "phase", "fit")
    protocol_path, protocol_sha = collection_protocol(phase)
    if sha256(protocol_path) != protocol_sha:
        raise ValueError("Logit-copy fit protocol changed")
    protocol = json.loads(protocol_path.read_text())
    identity = model_identity(args.model_id)
    synthetic = expected_sha is not None
    expected_sha = expected_sha or identity["sha256"]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "purpose": "Canonical CPU target rankings for predeclared development work replay; no speed claim",
        "collection_phase": phase,
        "synthetic_fixture": synthetic,
        "source": source_fingerprint(),
        "binary_sha256": sha256(args.binary),
        "model_id": args.model_id,
        "model_sha256": sha256(args.model),
        "protocol_sha256": protocol_sha,
        "platform": {"platform": platform.platform(), "machine": platform.machine()},
        "scope": (
            "Three development-fit prompts only; no validation or held-out data"
            if phase == "fit"
            else "Six predeclared development-validation prompts; no reserved held-out data"
        ),
    }
    atomic_json(out / "manifest.json", report)
    if report["model_sha256"] != expected_sha:
        raise ValueError("Wrong pinned model for canonical logit collection")
    corpus = {
        p["id"]: p for p in read_jsonl(corpus_path or ROOT / "bench/prompts-v1.jsonl")
    }
    jobs = [
        {
            "prompt_id": pid,
            "prompt": corpus[pid]["prompt"],
            "max_tokens": protocol["output_cap"],
        }
        for pid in protocol["prompt_ids"]
    ]
    write_jsonl(out / "jobs.jsonl", jobs)
    try:
        with (out / "collector.log").open("wb") as stream:
            subprocess.run(
                [
                    str(args.binary.resolve()),
                    str(args.model.resolve()),
                    str(out / "jobs.jsonl"),
                    str(out / "data.jsonl"),
                ],
                stdout=stream,
                stderr=stream,
                timeout=1200,
                check=True,
            )
        raw_lines = (out / "data.jsonl").read_text().splitlines()
        rows = [json.loads(line) for line in raw_lines]
        if len(rows) != len(jobs):
            raise ValueError("Missing canonical fit request")
        for job, row in zip(jobs, rows, strict=True):
            info = row["system_info"]
            if (
                row["prompt_id"] != job["prompt_id"]
                or row.get("observer_exact") is not True
                or "TOKENOMICS_LOGIT_COLLECTOR=ON" not in info
                or "TOKENOMICS_CPU_BATCH_INVARIANT=ON" not in info
                or any(
                    flag in info
                    for flag in (
                        "TOKENOMICS_GREEDY_NO_TEMP=ON",
                        "TOKENOMICS_GREEDY_DIRECT=ON",
                        "TOKENOMICS_CPU_Q4_PAIR=ON",
                        "TOKENOMICS_CPU_PACKED_VERIFY=ON",
                    )
                )
                or not row["prompt_ids"]
                or not 1 <= len(row["token_ids"]) <= protocol["output_cap"]
                or len(row["rows"]) != len(row["token_ids"])
            ):
                raise ValueError("Wrong canonical collector profile or coverage")
            for index, (token, logits) in enumerate(
                zip(row["token_ids"], row["rows"], strict=True)
            ):
                if (
                    logits["output_index"] != index
                    or len(logits["top_ids"]) != protocol.get("top_k_collected", 64)
                    or len(set(logits["top_ids"]))
                    != protocol.get("top_k_collected", 64)
                    or logits["top_ids"][0] != token
                ):
                    raise ValueError("Malformed canonical rank row")
        if (
            source_fingerprint() != report["source"]
            or sha256(args.binary) != report["binary_sha256"]
            or sha256(args.model) != report["model_sha256"]
            or sha256(protocol_path) != report["protocol_sha256"]
        ):
            raise ValueError(
                "Source, model, binary or protocol changed during collection"
            )
        status = {
            "state": "passed",
            "requests": len(rows),
            "ranking_rows": sum(len(r["rows"]) for r in rows),
            "synthetic_fixture": synthetic,
            "performance_claim": None,
        }
        atomic_json(out / "status.json", status)
        payloads = {
            name: sha256(out / name)
            for name in (
                "manifest.json",
                "status.json",
                "jobs.jsonl",
                "data.jsonl",
                "collector.log",
            )
        }
        atomic_json(out / "complete.json", {"sha256": payloads})
        # Fixed public development prompts only. Publish compressed numeric ranks
        # and selected hashes so artifact expiration cannot erase this small fit.
        for line, row in zip(raw_lines, rows, strict=True):
            raw = line.encode()
            item = {
                "prompt_id": row["prompt_id"],
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "gzip_base64": base64.b64encode(gzip.compress(raw, mtime=0)).decode(),
            }
            print("LOGIT_FIT_DATA=" + json.dumps(item, sort_keys=True), flush=True)
        receipt = {
            **report,
            **status,
            "payload_sha256": payloads,
            "complete_sha256": sha256(out / "complete.json"),
        }
        print("LOGIT_FIT_RECEIPT=" + json.dumps(receipt, sort_keys=True), flush=True)
        return status
    except Exception as error:
        atomic_json(
            out / "status.json",
            {
                "state": "failed",
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", choices=tuple(MODEL_PINS), default=DEFAULT_MODEL)
    parser.add_argument("--phase", choices=("fit", "validation"), default="fit")
    parser.add_argument("--out", type=Path, required=True)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
