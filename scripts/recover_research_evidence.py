"""Preserve allowlisted public research artifacts with their original ZIP digests.

Runs on a standard public runner; starts no model/build and prints only already
public diagnostic artifact bytes. Local import verifies every chunk and seal.
"""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import zipfile

REPO = "VictorWong123/tokenomics"
RUNS = {
    43848978979: ("trained-cpu-smoke", "3fdfa8c"),
    43849593799: ("trained-cpu-smoke", "1af7b53"),
    43861391090: ("canonical-logit-fit", "f9ecb4691306a8b1172b654136e58724e47432e4"),
    43861395334: ("canonical-logit-fit", "f9ecb4691306a8b1172b654136e58724e47432e4"),
}
LIMIT = 2 * 1024**2


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(data):
    if not 0 < len(data) <= LIMIT:
        raise ValueError("Oversized/empty ZIP")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        payload = {}
        size = 0
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or str(path) != item.filename.rstrip("/")
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError("Unsafe archive path")
            if item.is_dir():
                continue
            size += item.file_size
            if (
                path.suffix not in {".json", ".jsonl", ".log", ".gz"}
                or item.file_size > 16 * 1024**2
                or size > 64 * 1024**2
                or item.filename in payload
            ):
                raise ValueError("Unexpected/oversized/duplicate diagnostic payload")
            payload[item.filename] = archive.read(item)
    checked = {}
    for name, raw in payload.items():
        if PurePosixPath(name).name != "complete.json":
            continue
        seal = json.loads(raw)["sha256"]
        if not seal:
            raise ValueError("Empty payload seal")
        for relative, expected in seal.items():
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative:
                raise ValueError("Unsafe sealed path")
            full = str(PurePosixPath(name).parent / path)
            if full not in payload or digest(payload[full]) != expected:
                raise ValueError("Missing/mismatched sealed payload: " + full)
        checked[name] = {"sha256": digest(raw), "checked_payloads": len(seal)}
    if not checked:
        raise ValueError("No completed payload seal")
    return {"files": len(payload), "uncompressed_bytes": size, "seals": checked}


def api(path, *, binary=False):
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}/{path}"],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return result.stdout if binary else json.loads(result.stdout)


def export():
    for run_id, (name, revision) in RUNS.items():
        run = api(f"actions/runs/{run_id}")
        if (
            run["status"] != "completed"
            or run["conclusion"] != "success"
            or run["run_attempt"] != 1
            or not run["head_sha"].startswith(revision)
            or run["head_branch"] != "research/speculative-throughput-20261116"
        ):
            raise ValueError("Wrong public research run")
        artifacts = api(f"actions/runs/{run_id}/artifacts?per_page=100")["artifacts"]
        matches = [a for a in artifacts if a["name"] == name and not a["expired"]]
        if len(matches) != 1 or not 0 < matches[0]["size_in_bytes"] <= LIMIT:
            raise ValueError("Missing/ambiguous/oversized artifact")
        artifact = matches[0]
        data = api(f"actions/artifacts/{artifact['id']}/zip", binary=True)
        if artifact["digest"] != "sha256:" + digest(data):
            raise ValueError("GitHub ZIP digest mismatch")
        validation = validate(data)
        encoded = base64.b64encode(data).decode()
        chunks = [encoded[i : i + 16384] for i in range(0, len(encoded), 16384)]
        receipt = {
            "run_id": run_id,
            "run_url": run["html_url"],
            "source_commit": run["head_sha"],
            "conclusion": run["conclusion"],
            "artifact_id": artifact["id"],
            "artifact_name": name,
            "zip_sha256": digest(data),
            "bytes": len(data),
            "chunks": len(chunks),
            "validation": validation,
        }
        for i, chunk in enumerate(chunks):
            print(
                "RESEARCH_EVIDENCE_CHUNK="
                + json.dumps({"run_id": run_id, "index": i, "base64": chunk}),
                flush=True,
            )
        print(
            "RESEARCH_EVIDENCE_RECEIPT=" + json.dumps(receipt, sort_keys=True),
            flush=True,
        )


def import_log(log, out):
    receipts, chunks = {}, {}
    for line in Path(log).read_text().splitlines():
        if "RESEARCH_EVIDENCE_RECEIPT=" in line:
            item = json.loads(line.split("RESEARCH_EVIDENCE_RECEIPT=", 1)[1])
            run_id = item["run_id"]
            if run_id in receipts:
                raise ValueError("Duplicate artifact receipt")
            receipts[run_id] = item
        if "RESEARCH_EVIDENCE_CHUNK=" in line:
            item = json.loads(line.split("RESEARCH_EVIDENCE_CHUNK=", 1)[1])
            key = item["run_id"], item["index"]
            if key in chunks or not 0 <= item["index"] < 200:
                raise ValueError("Duplicate/oversized chunk set")
            if len(item["base64"]) > 16384:
                raise ValueError("Oversized chunk")
            chunks[key] = item["base64"]
    if set(receipts) != set(RUNS):
        raise ValueError("Missing/unexpected artifact receipt")
    archives = {}
    expected_keys = set()
    for run_id, item in receipts.items():
        name, revision = RUNS[run_id]
        if (
            item["artifact_name"] != name
            or not re.fullmatch(r"[0-9a-f]{40}", item["source_commit"])
            or not item["source_commit"].startswith(revision)
            or item["run_url"] != f"https://github.com/{REPO}/actions/runs/{run_id}"
            or item["conclusion"] != "success"
            or not 0 < item["chunks"] < 200
        ):
            raise ValueError("Wrong recovered run identity")
        keys = [(run_id, i) for i in range(item["chunks"])]
        expected_keys.update(keys)
        data = base64.b64decode("".join(chunks[key] for key in keys), validate=True)
        if len(data) != item["bytes"] or digest(data) != item["zip_sha256"]:
            raise ValueError("Recovered ZIP hash/size mismatch")
        if validate(data) != item["validation"]:
            raise ValueError("Recovered seal validation differs")
        archives[run_id] = data
    if set(chunks) != expected_keys:
        raise ValueError("Unexpected artifact chunks")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for run_id, data in archives.items():
        (out / f"{run_id}.zip").write_bytes(data)
        (out / f"{run_id}.json").write_text(
            json.dumps(receipts[run_id], indent=2) + "\n"
        )
    return receipts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.log and args.out:
        import_log(args.log, args.out)
    elif not args.log and not args.out:
        export()
    else:
        parser.error("Provide both --log/--out for import, or neither for export")
