#!/usr/bin/env python3
"""Terminal state-transition demo using synthetic fixtures, never a speedup demo."""
import argparse
import json
import pathlib
import subprocess
import tempfile
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench.common import read_jsonl

p = argparse.ArgumentParser()
p.add_argument("--build-dir", type=pathlib.Path, default=pathlib.Path("build-cpu"))
a = p.parse_args()
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
    print("cycle  k  actual  accepted  committed  draft catch-up tokens")
    for c in read_jsonl(d / "cycles.jsonl"):
        print(
            f"{c['cycle']:5d} {c['action']:2d} {c['actual_drafts']:7d} {c['accepted']:9d} {c['committed']:10d} {c['synced']:21d}"
        )
    r = read_jsonl(d / "requests.jsonl")[0]
    print(
        f"Committed {r['generated_tokens']} IDs; stop={r['stop']}; target cached={r['target_cached']}; draft cached={r['draft_cached']}"
    )
