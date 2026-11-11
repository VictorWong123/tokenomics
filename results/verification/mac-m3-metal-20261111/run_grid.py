"""Repeat the M3 0.5B development comparison from the repository root."""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))
from analysis.controller import plot_traces, report
from bench.common import ROOT, atomic_json, read_jsonl, sha256
from bench.controller import calibrate, configurations
from bench.replay import investigate
from bench.run import run


def main(out):
    binary = ROOT / "build-metal/tokenomics"
    model = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    lock = json.loads(
        (
            ROOT / "results/verification/codespace-gate-20261108/models.lock.json"
        ).read_text()
    )
    if sha256(model) != lock["models"]["qwen25-05"]["sha256"]:
        raise ValueError("Unexpected target model")
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 1800
    cfg = json.loads((ROOT / "configs/controller-study-v1.json").read_text())
    cfg["target_model"] = "qwen25-05"
    prompts = read_jsonl(ROOT / "bench/prompts-v1.jsonl")

    def measure(name, selected, config):
        return run(
            binary,
            model,
            None,
            [p for p in prompts if p["id"] in selected],
            config,
            out / name,
            gpu_layers=99,
            deadline=deadline,
            identity={"synthetic_fixture": False, "purpose": config["phase"]},
        )

    fit = dict(
        cfg,
        phase="mac_m3_05b_metal_fit",
        output_caps=cfg["fit_output_caps"],
        repetitions=cfg["fit_repetitions"],
        configurations=[{"id": "target", "policy": "target"}]
        + [
            {"id": f"fixed{k}", "policy": "fixed", "k": k, "drafter": "ngram"}
            for k in (1, 2, 4, 6, 8)
        ],
    )
    measure("fit", cfg["fit_prompt_ids"], fit)
    investigate(
        binary,
        model,
        None,
        out / "fit",
        out / "replay",
        fit,
        gpu_layers=99,
        drafter="ngram",
        deadline=deadline,
    )
    prior = calibrate(out / "fit", out / "replay", "ngram")
    atomic_json(out / "prior.json", prior)
    cfg.update(
        phase="mac_m3_05b_metal_validation",
        selected_fixed=prior["provenance"]["selected_fixed"],
        development_selection_sha256=sha256(out / "prior.json"),
        configurations=configurations(prior),
    )
    measure("validation", cfg["validation_prompt_ids"], cfg)
    report(out / "validation")
    measure_logging(out)


def measure_logging(out):
    """Repeat the median/worst-prompt logging grid used in the Linux study."""
    validation = out / "validation"
    frozen = json.loads((validation / "run_manifest.json").read_text())["frozen"]
    summary = json.loads((validation / "analysis/summary.json").read_text())
    cfg = frozen["config"]
    selected = set(summary["trace_prompt_selection"].values())
    conditions = [
        c
        for c in cfg["configurations"]
        if c["id"] in ("target", cfg["selected_fixed"], "cost-aware")
    ]
    controller = next(c for c in conditions if c["id"] == "cost-aware")
    cfg.update(
        phase="controller_logging_diagnostic",
        output_caps=[max(cfg["output_caps"])],
        repetitions=2,
        configurations=conditions
        + [dict(controller, id="cost-aware-detailed", detailed=True)],
    )
    run(
        ROOT / "build-metal/tokenomics",
        ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        None,
        [p for p in frozen["prompts"] if p["id"] in selected],
        cfg,
        out / "logging",
        gpu_layers=99,
        deadline=time.monotonic() + 300,
        identity={"synthetic_fixture": False, "purpose": cfg["phase"]},
    )
    report(out / "logging")
    plot_traces(out / "logging")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python run_grid.py NEW_OUTPUT_DIRECTORY")
    main(Path(sys.argv[1]))
