"""Audit preserved public CPU research records without building or inference."""

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import zipfile

from bench.common import atomic_json
from bench.model_identity import model_identity
from scripts.check_metal_profile import check_direct_execution
from scripts.check_models import check_cycles, compare
from scripts.recover_research_evidence import RUNS, digest, validate

WORK = ("cycles", "drafted", "accepted", "verified_rows")


def key(row):
    return row["prompt_id"], row["configuration_id"], row["repetition"]


def audit_cycles(rows, cycles):
    check_cycles(rows, cycles)
    grouped = defaultdict(list)
    for cycle in cycles:
        grouped[key(cycle)].append(cycle)
    for row in rows:
        records = grouped[key(row)]
        counts = {
            "cycles": len(records),
            "drafted": sum(c["actual_drafts"] for c in records),
            "accepted": sum(c["accepted"] for c in records),
            "verified_rows": sum(1 + c["actual_drafts"] for c in records),
        }
        if any(row["execution"][name] != value for name, value in counts.items()) or [
            c["cycle"] for c in records
        ] != list(range(len(records))):
            raise ValueError("Cycle work/index mismatch")
        if row["settings"]["drafter"] != "logit-copy":
            continue
        cursor = 0
        sums = {
            name: 0 for name in ("source_candidates", "rank_checks", "logits_examined")
        }
        for cycle in records:
            trace = cycle["logit_copy"]
            expected_index = cursor - 1 if trace["hint_available"] else -1
            if trace["hint_output_index"] != expected_index or (
                trace["hint_available"] and cursor == 0
            ):
                raise ValueError("Future/stale canonical logit hint")
            for name in sums:
                sums[name] += trace[name]
            cursor += cycle["committed"]
        if any(row["logit_copy"][name] != value for name, value in sums.items()):
            raise ValueError("Guided telemetry sum mismatch")
    return len(cycles)


def audit(directory, out):
    directory = Path(directory)
    reports = {}
    for run_id, (artifact_name, revision) in RUNS.items():
        meta = json.loads((directory / f"{run_id}.json").read_text())
        data = (directory / f"{run_id}.zip").read_bytes()
        if (
            digest(data) != meta["zip_sha256"]
            or len(data) != meta["bytes"]
            or validate(data) != meta["validation"]
            or meta["artifact_name"] != artifact_name
            or meta["run_id"] != run_id
            or not meta["source_commit"].startswith(revision)
        ):
            raise ValueError("Preserved artifact identity/hash differs")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:

            def document(name):
                return json.loads(archive.read(name))

            def records(name):
                return [
                    json.loads(line)
                    for line in archive.read(name).splitlines()
                    if line.strip()
                ]

            direct = artifact_name == "trained-cpu-smoke"
            prefix = "smoke/" if direct else "live/"
            manifest = document(prefix + "manifest.json")
            status = document(prefix + "status.json")
            if (
                manifest["synthetic_fixture"]
                or manifest["source"]["revision"] != meta["source_commit"]
                or manifest["model_sha256"]
                != model_identity(manifest["model_id"])["sha256"]
                or status["state"] != "passed"
                or status["synthetic_fixture"]
            ):
                raise ValueError("Wrong learned source/model/result")
            comparisons = 0
            cycle_count = 0
            if direct:
                expected = {
                    row["prompt_id"]: row
                    for name in ("reference", "boundary-reference")
                    for row in records(prefix + name + ".jsonl")
                }
                stages = {}
                for stage in ("guarded", "greedy"):
                    rows = records(prefix + stage + ".jsonl")
                    if len(rows) != 80:
                        raise ValueError("Incomplete trained sampler grid")
                    cycle_count += audit_cycles(
                        rows, records(prefix + stage + ".cycles.jsonl")
                    )
                    for row in rows:
                        if compare(expected[row["prompt_id"]], row) is not None:
                            raise ValueError("Retained sampler output differs")
                        is_direct = "TOKENOMICS_GREEDY_DIRECT=ON" in row["system_info"]
                        if is_direct != (stage == "greedy"):
                            raise ValueError("Wrong retained sampler profile")
                        if is_direct:
                            check_direct_execution(row)
                        comparisons += 1
                    stages[stage] = {key(row): row for row in rows}
                if stages["guarded"].keys() != stages["greedy"].keys():
                    raise ValueError("Sampler grids differ")
                for request, row in stages["guarded"].items():
                    if any(
                        row["execution"][name]
                        != stages["greedy"][request]["execution"][name]
                        for name in (*WORK, "synced_tokens")
                    ):
                        raise ValueError("Sampler work differs")
                isas = sorted(
                    {
                        row["system_info"].split("CPU : ", 1)[-1].split(";", 1)[0]
                        for row in rows
                    }
                )
            else:
                expected = {row["prompt_id"]: row for row in records("data.jsonl")}
                replay = document(prefix + "canonical-replay/fit.json")
                predicted = {
                    (
                        row["prompt_id"],
                        row["configuration"]["id"] + f"-k{row['configuration']['cap']}",
                    ): row
                    for row in replay["per_prompt"]
                }
                rows = records(prefix + "live.jsonl")
                if len(rows) != 60 or len(predicted) != 60:
                    raise ValueError("Incomplete live validation grid")
                cycle_count += audit_cycles(rows, records(prefix + "live.cycles.jsonl"))
                for row in rows:
                    wanted = predicted[(row["prompt_id"], row["configuration_id"])]
                    if compare(expected[row["prompt_id"]], row) is not None or any(
                        row["execution"][name] != wanted[name] for name in WORK
                    ):
                        raise ValueError("Retained live output/work differs")
                    comparisons += 1
                isas = sorted(
                    {
                        row["system_info"]
                        .split("logit_copy_rank_isa=", 1)[1]
                        .split(";", 1)[0]
                        for row in rows
                    }
                )
            if comparisons != status["checked_comparisons"]:
                raise ValueError("Retained comparison count differs")
            reports[str(run_id)] = {
                "state": "passed",
                "model_id": manifest["model_id"],
                "source_commit": meta["source_commit"],
                "zip_sha256": digest(data),
                "checked_comparisons": comparisons,
                "audited_cycles": cycle_count,
                "instruction_set_markers": isas,
                "performance_claim": None,
            }
    result = {
        "scope": "Offline recheck of original GitHub ZIP hashes/seals, learned IDs/stops, per-cycle work and causal guided hints; no rerun or speed claim",
        "runs": reports,
    }
    atomic_json(Path(out), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.out), indent=2))
