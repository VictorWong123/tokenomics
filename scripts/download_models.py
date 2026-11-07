#!/usr/bin/env python3
"""Hub revision + LFS SHA pinning; resumes verified downloads without Git blobs."""
import argparse
import json
import pathlib
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench.common import ROOT, atomic_json, sha256


def download(model_id, destination, lock_path):
    catalog = json.loads((ROOT / "configs/models-v1.json").read_text())["models"]
    spec = catalog[model_id]
    lock = (
        json.loads(lock_path.read_text())
        if lock_path.exists()
        else {"schema_version": 1, "models": {}}
    )
    if model_id not in lock["models"]:
        url = f"https://huggingface.co/api/models/{spec['repo']}?blobs=true"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                info = json.load(response)
        except Exception as ex:
            raise RuntimeError(
                f"Cannot resolve {url}: {ex}. Allow huggingface.co and its redirected *.hf.co/*.huggingface.co hosts; rerun unchanged. No model results exist yet."
            ) from ex
        matches = [x for x in info["siblings"] if x["rfilename"] == spec["file"]]
        if len(matches) != 1 or "lfs" not in matches[0]:
            raise RuntimeError(
                f"Missing exact LFS artifact {spec['file']}; inspect upstream naming, do not guess weights"
            )
        entry = {
            **spec,
            "revision": info["sha"],
            "sha256": matches[0]["lfs"]["sha256"],
            "bytes": matches[0]["lfs"]["size"],
        }
        lock["models"][model_id] = entry
        atomic_json(lock_path, lock)
    entry = lock["models"][model_id]
    if entry["repo"] != spec["repo"] or entry["file"] != spec["file"]:
        raise RuntimeError("model lock conflicts with catalog")
    destination.mkdir(parents=True, exist_ok=True)
    out = destination / entry["file"]
    if out.exists():
        if sha256(out) == entry["sha256"]:
            return out
        raise RuntimeError(f"Existing {out} hash mismatch; preserve it and investigate")
    url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{entry['file']}"
    tmp = out.with_suffix(out.suffix + ".part")
    # curl uses inherited proxy and CA trust; no bypass route or embedded token.
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "3",
            "--connect-timeout",
            "30",
            "--continue-at",
            "-",
            "--output",
            str(tmp),
            url,
        ],
        check=True,
    )
    if tmp.stat().st_size != entry["bytes"] or sha256(tmp) != entry["sha256"]:
        raise RuntimeError(f"download hash/size mismatch: {tmp}")
    tmp.replace(out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--directory", type=pathlib.Path, default=ROOT / "models")
    p.add_argument(
        "--lock", type=pathlib.Path, default=ROOT / "models/models.lock.json"
    )
    a = p.parse_args()
    try:
        print(download(a.model, a.directory, a.lock))
    except Exception as ex:
        print(str(ex), file=sys.stderr)
        sys.exit(2)
