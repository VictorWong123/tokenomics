#!/usr/bin/env python3
"""Show synthetic transitions or a verified recorded trained-model trace."""
import argparse
import json
import pathlib
import subprocess
import tempfile
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench.common import read_jsonl, sha256
from scripts.check_models import compare, check_cycles


def print_trace(cycles, row):
    print("cycle  k  actual  accepted  committed  draft catch-up tokens")
    for c in cycles:
        print(
            f"{c['cycle']:5d} {c['action']:2d} {c['actual_drafts']:7d} {c['accepted']:9d} {c['committed']:10d} {c['synced']:21d}"
        )
    print(
        f"Committed {row['generated_tokens']} IDs; stop={row['stop']}; target cached={row['target_cached']}; draft cached={row['draft_cached']}"
    )


def recorded_demo(root):
    manifest = json.loads((root / "execution-manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError("Unsafe recorded evidence path")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError("Recorded evidence hash changed")
    gate = root / "gate"
    rows = read_jsonl(gate / "candidate.jsonl")
    matches = [
        r
        for r in rows
        if (r["prompt_id"], r["configuration_id"], r["repetition"])
        == ("development-qa-00", "retain-script", 0)
    ]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous recorded demo request")
    row = matches[0]
    refs = [
        r
        for r in read_jsonl(gate / "reference.jsonl")
        if r["prompt_id"] == row["prompt_id"]
    ]
    if len(refs) != 1 or compare(refs[0], row) is not None:
        raise ValueError("Recorded demo differs from ordinary target IDs")
    cycles = [
        c
        for c in read_jsonl(gate / "candidate.cycles.jsonl")
        if (c["prompt_id"], c["configuration_id"], c["repetition"])
        == (row["prompt_id"], row["configuration_id"], row["repetition"])
    ]
    check_cycles([row], cycles)
    print(
        "RECORDED TRAINED-QWEN TRACE — scripted correctness demo; not cost-aware scheduling or a speedup claim"
    )
    print(f"Run {manifest['run_id']}; source {manifest['sha']}")
    print("Original artifact file hashes and ordinary target token IDs verified.")
    print_trace(cycles, row)


p = argparse.ArgumentParser()
p.add_argument("--build-dir", type=pathlib.Path, default=pathlib.Path("build-cpu"))
p.add_argument(
    "--recorded",
    type=pathlib.Path,
    nargs="?",
    const=pathlib.Path(__file__).resolve().parents[1]
    / "results/verification/codespace-gate-20261108",
    help="Replay the checked-in trained trace (or supply its evidence directory); no inference",
)
a = p.parse_args()
if a.recorded is not None:
    recorded_demo(a.recorded)
    sys.exit(0)
b = a.build_dir.resolve()
with tempfile.TemporaryDirectory() as temp:
    d = pathlib.Path(temp)
    job = d / "jobs.jsonl"
    job.write_text(
        json.dumps(
            {
                "run_id": "synthetic-state-demo",
                "configuration_id": "script",
                "prompt_id": "fixture-a",
                "repetition": 0,
                "prompt": "a b",
                "policy": "script",
                "script": [0, 0, 4, 1, 0, 8, 2],
                "max_tokens": 40,
            }
        )
        + "\n"
    )
    subprocess.run(
        [
            str(b / "tokenomics"),
            "--target",
            str(b / "fixtures/target.gguf"),
            "--draft",
            str(b / "fixtures/target.gguf"),
            "--jobs",
            str(job),
            "--out",
            str(d / "requests.jsonl"),
            "--cycles",
            str(d / "cycles.jsonl"),
            "--threads",
            "1",
            "--context",
            "128",
        ],
        check=True,
    )
    print("SYNTHETIC TRANSFORMER STATE DEMO — not a language-model speedup claim")
    r = read_jsonl(d / "requests.jsonl")[0]
    print_trace(read_jsonl(d / "cycles.jsonl"), r)
