#!/usr/bin/env python3
"""Check production C++ suffix proposals on archived canonical validation IDs.

This executes retrieval only, without model weights or inference.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.controller import verify
from analysis.proposal_validation import evaluate
from bench.common import (
    ROOT,
    atomic_json,
    read_jsonl,
    sha256,
    source_fingerprint,
    write_jsonl,
)


def jobs_for(directory, label, frozen_result):
    verify(directory)
    current = evaluate(directory, label)
    for key in ("totals", "cells", "comparisons", "requests_sha256", "manifest_sha256"):
        if current[key] != frozen_result[key]:
            raise ValueError(
                "Development validation replay differs from frozen evidence"
            )
    expected = {
        (row["prompt_id"], row["output_cap"]): row
        for row in frozen_result["cells"]
        if row["strategy"] == "suffix5-cap1"
    }
    jobs = []
    for row in read_jsonl(Path(directory) / "requests.jsonl"):
        if row["configuration_id"] != "target" or row["repetition"] != 0:
            continue
        counts = expected[(row["prompt_id"], row["output_cap"])]
        jobs.append(
            {
                "source_id": label,
                "prompt_id": row["prompt_id"],
                "output_cap": row["output_cap"],
                "prompt_ids": row["prompt_ids"],
                "token_ids": row["token_ids"],
                "cap": 1,
                "expected": {
                    key: counts[key]
                    for key in (
                        "tokens",
                        "cycles",
                        "drafted",
                        "accepted",
                        "verified_rows",
                        "wasted_rows",
                    )
                },
            }
        )
    if len(jobs) != 12:
        raise ValueError("Expected twelve native replay cells per platform")
    return jobs


def check(binary, out):
    binary = Path(binary).resolve()
    binary_hash = sha256(binary)
    source = source_fingerprint()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(
        (
            ROOT
            / "results/verification/suffix-validation-replay-20261116/validation.json"
        ).read_text()
    )
    sources = {source["label"]: source for source in frozen["sources"]}
    jobs = jobs_for(
        ROOT
        / "results/verification/codespace-controller-20261108/attempt-2/validation",
        "linux-cpu-3b",
        sources["linux-cpu-3b"],
    )
    archive = ROOT / "results/verification/mac-m3-metal-20261111/validation.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        seen = set()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                path = PurePosixPath(member.name)
                if not path.parts or path.parts[0] != "validation" or member.isdir():
                    continue
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not member.isfile()
                    or member.name in seen
                ):
                    raise ValueError("Unsafe archive member")
                seen.add(member.name)
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
        jobs += jobs_for(root / "validation", "m3-metal-05b", sources["m3-metal-05b"])
    write_jsonl(out / "jobs.jsonl", jobs)
    with (out / "native.jsonl").open("w") as stdout, (out / "stderr.log").open(
        "w"
    ) as stderr:
        subprocess.run(
            [str(Path(binary).resolve()), str((out / "jobs.jsonl").resolve())],
            stdout=stdout,
            stderr=stderr,
            check=True,
            timeout=60,
        )
    rows = read_jsonl(out / "native.jsonl")
    if len(rows) != 24:
        raise ValueError("Incomplete native replay")
    for job, row in zip(jobs, rows):
        if any(
            row.get(key) != job[key] for key in ("source_id", "prompt_id", "output_cap")
        ) or any(row.get(key) != value for key, value in job["expected"].items()):
            raise ValueError("Native replay identity/count mismatch")
    if sha256(binary) != binary_hash or source_fingerprint() != source:
        raise ValueError("Native replay executable or source changed during the check")
    result = {
        "schema_version": 1,
        "state": "native_suffix_validation_replay_passed",
        "source": source,
        "binary_sha256": binary_hash,
        "cases": 24,
        "payload_sha256": {
            name: sha256(out / name)
            for name in ("jobs.jsonl", "native.jsonl", "stderr.log")
        },
        "frozen_validation_sha256": sha256(
            ROOT
            / "results/verification/suffix-validation-replay-20261116/validation.json"
        ),
        "model_inference_executed": False,
        "limitation": "Actual production C++ proposals replayed against archived canonical IDs. This validates proposal work counts, not alternate-path target arithmetic or throughput.",
    }
    atomic_json(out / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.binary, args.out), indent=2))


if __name__ == "__main__":
    main()
