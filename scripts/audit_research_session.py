"""Recheck retained CPU proposal, sampler and projection evidence offline."""

import argparse
import contextlib
import gzip
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.logit_copy import analyze as guided_replay
from analysis.matvec_component import SHAPES_BY_MODEL, encoded_weight_bytes
from analysis.research_archive import audit as raw_audit
from analysis.transition_replay import analyze as exploratory_replay
from bench.common import ROOT, atomic_json, sha256, source_fingerprint
from bench.model_identity import MODEL_PINS

EVIDENCE = ROOT / "results/verification"


def sealed(directory, expected):
    seal = json.loads((directory / "complete.json").read_text())["sha256"]
    if set(seal) != set(expected) or any(
        sha256(directory / name) != value for name, value in seal.items()
    ):
        raise ValueError("Retained payload changed: " + str(directory))


def check_projection_census():
    root = EVIDENCE / "projection-census-20261116"
    sealed(root, ["qwen25-05.json", "qwen25-15.json", "summary.json"])
    summary = json.loads((root / "summary.json").read_text())["models"]
    for model, shapes in SHAPES_BY_MODEL.items():
        header = json.loads((root / (model + ".json")).read_text())
        census = header["projection_census"]
        if (
            header["lock"] != MODEL_PINS[model]
            or header["weights_saved"]
            or header["full_model_sha256_verified"]
        ):
            raise ValueError("Header scope/model changed")
        projected = covered = 0
        for role in census["projection_by_role"].values():
            total = 0
            for variant in role["variants"]:
                shape = (variant["type"].lower().replace("_k", "_K"), *variant["shape"])
                value = encoded_weight_bytes(*shape) * variant["tensors"]
                if value != variant["encoded_bytes"]:
                    raise ValueError("Encoded projection size differs")
                total += value
                if shape in shapes:
                    covered += value
            if total != role["encoded_bytes"]:
                raise ValueError("Projection role does not reconcile")
            projected += total
        if (
            projected != census["projection_bytes"]
            or projected
            != sum(
                row["encoded_bytes"] for row in census["projection_by_type"].values()
            )
            or projected
            + census["embedding_tensor_bytes"]
            + census["other_tensor_bytes"]
            != census["total_encoded_tensor_bytes"]
            or ((header["header_bytes_parsed"] + 31) // 32) * 32
            + census["total_encoded_tensor_bytes"]
            != header["lock"]["bytes"]
            or covered != summary[model]["selected_shape_class_covered_bytes"]
            or covered / projected != summary[model]["selected_shape_class_fraction"]
            or covered / projected <= 0.98
        ):
            raise ValueError("Projection census or component coverage differs")


def audit(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    source = source_fingerprint()
    report = {
        "state": "running",
        "source": source,
        "scope": "Offline seal/raw-record/canonical-work regression audit; starts no inference, downloads or GPU work",
        "collections": [],
        "performance_claim": None,
    }
    try:
        with tempfile.TemporaryDirectory(dir=out) as tmp, (out / "replay.log").open(
            "w"
        ) as replay_log:
            temporary = Path(tmp)
            raw = raw_audit(
                EVIDENCE / "research-raw-20261116", temporary / "raw-audit.json"
            )
            if raw != json.loads(
                (EVIDENCE / "research-raw-20261116/offline-audit.json").read_text()
            ):
                raise ValueError("Original learned CPU audit differs")
            report["learned_comparisons"] = sum(
                row["checked_comparisons"] for row in raw["runs"].values()
            )
            report["measured_cycles"] = sum(
                row["audited_cycles"] for row in raw["runs"].values()
            )
            check_projection_census()
            report["projection_census"] = "passed"
            for collection in sorted(
                EVIDENCE.glob("logit-copy-*-20261116/*/collection")
            ):
                if collection.parent.parent.name not in {
                    "logit-copy-fit-20261116",
                    "logit-copy-validation-20261116",
                    "logit-copy-live-20261116",
                }:
                    continue
                prior = collection.parent / "replay"
                sealed(prior, ["fit.json", "cycles.jsonl.gz"])
                destination = temporary / (
                    collection.parent.parent.name + "-" + collection.parent.name
                )
                with contextlib.redirect_stdout(replay_log):
                    guided_replay(collection, destination)
                current = json.loads((destination / "fit.json").read_text())
                retained = json.loads((prior / "fit.json").read_text())
                for field in ("totals", "per_prompt", "work_frontier"):
                    if current[field] != retained[field]:
                        raise ValueError(
                            "Guided replay differs: " + str(collection) + " " + field
                        )
                if gzip.decompress(
                    (prior / "cycles.jsonl.gz").read_bytes()
                ) != gzip.decompress((destination / "cycles.jsonl.gz").read_bytes()):
                    raise ValueError("Guided per-cycle proposals/work/hints differ")
                report["collections"].append(str(collection.relative_to(EVIDENCE)))
            for family, folder in (
                ("transition", "transition-replay-20261116"),
                ("priority", "retrieval-priority-20261116"),
            ):
                for prior in sorted((EVIDENCE / folder).glob("*/report.json")):
                    sealed(prior.parent, ["report.json", "cycles.jsonl.gz"])
                    label = prior.parent.name
                    base = (
                        "logit-copy-fit-20261116"
                        if label.startswith("fit-")
                        else (
                            "logit-copy-live-20261116"
                            if "-arm-" in label
                            else "logit-copy-validation-20261116"
                        )
                    )
                    collection = (
                        EVIDENCE / base / ("qwen25-" + label[-2:]) / "collection"
                    )
                    destination = temporary / (folder + "-" + label)
                    with contextlib.redirect_stdout(replay_log):
                        exploratory_replay(collection, destination, family)
                    current = json.loads((destination / "report.json").read_text())
                    retained = json.loads(prior.read_text())
                    for field in ("totals", "per_prompt", "work_frontier"):
                        if current[field] != retained[field]:
                            raise ValueError(
                                "Exploratory replay differs: "
                                + str(prior)
                                + " "
                                + field
                            )
                    if gzip.decompress(
                        (prior.parent / "cycles.jsonl.gz").read_bytes()
                    ) != gzip.decompress(
                        (destination / "cycles.jsonl.gz").read_bytes()
                    ):
                        raise ValueError(
                            "Exploratory per-cycle proposals/work/hints differ"
                        )
                    report["collections"].append(
                        str(prior.parent.relative_to(EVIDENCE))
                    )
            if len(report["collections"]) != 16 or source_fingerprint() != source:
                raise ValueError("Incomplete replay coverage or source changed")
        report["state"] = "passed"
        atomic_json(out / "status.json", report)
        return report
    except Exception as error:
        report.update(state="failed", error=str(error), error_type=type(error).__name__)
        atomic_json(out / "status.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.out)
    print(json.dumps({**result, "collections": len(result["collections"])}, indent=2))
