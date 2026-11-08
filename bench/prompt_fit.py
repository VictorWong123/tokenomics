"""Hash-checked tokenizer-only context preflight; never generates held-out answers."""

from __future__ import annotations
import json
from pathlib import Path
import subprocess
from bench.common import atomic_json, read_jsonl, sha256


def check(exe, target, prompts, cfg, out, gpu_layers=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(
        binary_sha256=sha256(exe),
        target_sha256=sha256(target),
        prompts=prompts,
        context=cfg["context"],
        batch=cfg["batch"],
        threads=cfg["threads"],
        gpu_layers=gpu_layers,
        output_caps=cfg["output_caps"],
        tokenization_only=True,
    )
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Prompt-fit preflight inputs changed; use a new run directory")
    atomic_json(manifest, identity)
    receipt = out / "complete.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        record = out / previous["records"]
        if sha256(record) != previous["sha256"]:
            raise ValueError("Prompt-fit tokenization artifact changed")
    else:
        attempt = out / f"attempt-{len(list(out.glob('attempt-*')))}"
        attempt.mkdir()
        jobs = attempt / "jobs.jsonl"
        jobs.write_text(
            "".join(
                json.dumps(
                    dict(
                        prompt_id=p["id"],
                        prompt=p["prompt"],
                        policy="target",
                        max_tokens=0,
                        detailed=False,
                    )
                )
                + "\n"
                for p in prompts
            )
        )
        record = attempt / "requests.jsonl"
        command = [
            str(exe),
            "--target",
            str(target),
            "--jobs",
            str(jobs),
            "--out",
            str(record),
            "--context",
            str(cfg["context"]),
            "--batch",
            str(cfg["batch"]),
            "--threads",
            str(cfg["threads"]),
            "--gpu-layers",
            str(gpu_layers),
        ]
        with (attempt / "stderr.txt").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=log, timeout=300)
    rows = read_jsonl(record)
    if len(rows) != len(prompts) or any(
        r["prompt_id"] != p["id"] or r["generated_tokens"] != 0 or r["token_ids"]
        for r, p in zip(rows, prompts)
    ):
        raise ValueError("Incomplete tokenizer-only prompt preflight")
    if any(
        len(r["prompt_ids"]) + max(cfg["output_caps"]) - 1 > cfg["context"]
        for r in rows
    ):
        raise ValueError(
            "Frozen prompt/output caps exceed context; revise before evaluation"
        )
    if not receipt.exists():
        atomic_json(
            receipt,
            dict(
                records=record.relative_to(out).as_posix(),
                sha256=sha256(record),
                prompts=len(rows),
                max_prompt_tokens=max(len(r["prompt_ids"]) for r in rows),
            ),
        )
    return rows
