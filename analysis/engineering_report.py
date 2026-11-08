"""Generate a four-page engineering checkpoint from independently checked evidence."""

import argparse
import datetime as dt
import json
from pathlib import Path
import textwrap

from bench.common import ROOT, atomic_json, sha256
from scripts.demo_controller import recorded_directory
from scripts.verify_controller_evidence import check, equal


def generate(out, profile_receipt=None, preview_dir=None):
    raw = recorded_directory()
    canonical = raw.parent / "verification.json"
    receipt = json.loads(canonical.read_text())
    audited = check(raw)
    if not equal(audited["controller"], receipt["controller"]):
        raise ValueError("Controller receipt differs from regenerated measurements")
    result = audited["controller"]
    comparisons = result["comparisons"]
    costs = result["stage_costs"]["cost-aware"]
    controller = costs["controller_totals"]
    profile = None
    if profile_receipt is not None:
        from scripts.verify_profile_evidence import check as check_profiles

        proof = json.loads(profile_receipt.read_text())
        if (
            proof["state"] != "profile_comparison_complete"
            or not proof["independent_output_checks"]
            or not proof["shutdown_confirmed"]
        ):
            raise ValueError(
                "A complete independently verified CPU comparison is required"
            )
        point = profile_receipt.parent / f"attempt-{proof['attempt']}"
        regenerated = check_profiles(point, proof["source_sha"])
        if not equal(regenerated["paired_profile"], proof["paired_profile"]):
            raise ValueError(
                "CPU comparison receipt differs from regenerated measurements"
            )
        profile = regenerated["paired_profile"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
    output = out / "tokenomics-engineering-report.pdf"
    page_size = (8.27, 11.69)
    pages = []

    def page(number, title):
        fig = plt.figure(figsize=page_size, facecolor="white")
        fig.text(
            0.07,
            0.961,
            "TOKENOMICS  /  ENGINEERING CHECKPOINT  /  8 NOVEMBER 2026",
            size=8,
            color="#475569",
        )
        fig.text(0.07, 0.93, title, size=19, weight="bold", va="top", color="#0f172a")
        fig.text(
            0.07,
            0.031,
            "Linux CPU development evidence; full held-out and Mac/Metal results remain unexecuted.",
            size=7.5,
            color="#475569",
        )
        fig.text(0.93, 0.031, f"{number} / 4", size=8, ha="right", color="#475569")
        pages.append(fig)
        return fig

    def paragraph(fig, text, y, *, size=10, width=96, gap=0.012, color="#1e293b"):
        lines = []
        for part in text.split("\n"):
            lines.extend(
                textwrap.wrap(part, width=width, break_long_words=False) or [""]
            )
        fig.text(
            0.07,
            y,
            "\n".join(lines),
            size=size,
            va="top",
            linespacing=1.35,
            color=color,
        )
        height = len(lines) * size / 72 * 1.35 / page_size[1]
        return y - height - gap

    def heading(fig, text, y):
        fig.text(0.07, y, text, size=12, weight="bold", va="top", color="#0f172a")
        return y - 0.031

    def table(fig, columns, rows, box, widths=None, size=9):
        ax = fig.add_axes(box)
        ax.axis("off")
        t = ax.table(
            cellText=rows,
            colLabels=columns,
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=widths,
        )
        t.auto_set_font_size(False)
        t.set_fontsize(size)
        t.scale(1, 1.6)
        for (row, _), cell in t.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_facecolor("#e2e8f0")
                cell.set_text_props(weight="bold", color="#0f172a")
        return t

    def metric(name):
        m = comparisons["cost-aware/" + name]
        return f"{m['ratio']:.4f}", f"[{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}]"

    fig = page(1, "C++ runtime contribution and scope")
    y = paragraph(
        fig,
        "Tokenomics implements a single-request C++20 speculative-decoding runtime around a pinned llama.cpp. "
        "It preserves upstream greedy verification while owning policy decisions, cache reconciliation, "
        "retained draft state, lazy catch-up, target-only fallback and request reset.",
        0.877,
    )
    y = heading(fig, "Implemented contribution", y)
    y = paragraph(
        fig,
        "The experimental cost-aware controller selects actions {0, 1, 2, 4, 6, 8} using measured "
        "committed tokens per millisecond. Fixed arrays keep its valid select/observe path allocation-free. "
        "Calibration, model/binary identity, immutable comparison blocks, independent verification and "
        "minimal/detailed telemetry make the result inspectable and reproducible.",
        y,
    )
    y = heading(fig, "Measured result", y)
    y = paragraph(
        fig,
        f"The completed study has 252 exact validation records: six development prompts, caps 64/128, "
        f"three repetitions and seven conditions. Cost-aware throughput is {metric('dynamic')[0]}x cap-8 ngram, "
        f"{metric('fixed1')[0]}x the development-frozen fixed-1 baseline and {metric('target')[0]}x ordinary "
        "target-only decoding. It does not establish ordinary-decoding acceleration.",
        y,
    )
    baseline = result["baseline_table"]
    y = paragraph(
        fig,
        f"Cost-aware achieves {baseline['cost-aware']['geomean_median_tps']:.3f} tokens/s. "
        f"Median warm completion is {baseline['cost-aware']['median_warm_ms']/1000:.2f}s "
        f"versus {baseline['target']['median_warm_ms']/1000:.2f}s for ordinary decoding. "
        f"Peak lifetime RSS is {baseline['cost-aware']['peak_lifetime_rss_bytes']/2**30:.2f} "
        f"versus {baseline['target']['peak_lifetime_rss_bytes']/2**30:.2f} GiB, including load/warm-up.",
        y,
        size=9.5,
    )
    table(
        fig,
        [
            "Baseline",
            "Decode t/s",
            "Cost-aware / baseline (95% CI)",
        ],
        [
            [name, f"{baseline[key]['geomean_median_tps']:.3f}", " ".join(metric(key))]
            for name, key in [
                ("Ordinary target", "target"),
                ("Frozen fixed-1", "fixed1"),
                ("Ngram cap-8", "dynamic"),
                ("Acceptance heuristic", "acceptance"),
            ]
        ],
        [0.07, 0.35, 0.86, 0.18],
        [0.35, 0.16, 0.49],
    )
    y = heading(fig, "Operating point and separation", 0.322)
    y = paragraph(
        fig,
        "Qwen2.5-3B-Instruct Q4_K_M with ngram-simple drafting; context 2048, batch 512, greedy "
        "temperature-only sampling, F16 KV and two inference threads. The allocated Linux Codespace has "
        "two logical CPUs, reports an AMD EPYC 7763 topology and about 7.8 GiB RAM, with no swap or GPU. "
        "The three calibration prompts are separate from validation. Rates are geometric means of cell "
        "median throughput. Original synthetic CC0 prompts are a controlled workload, not a quality benchmark.",
        y,
        size=9.5,
    )
    paragraph(
        fig,
        "The controller was completed experimentally despite negative feasibility evidence. "
        "This checkpoint is not full Pilot A/B, the 90-prompt held-out evaluation, a Mac result or a novelty claim. "
        "A separate CPU-profile experiment cannot alter this frozen guarded-profile result.",
        y,
        size=9.5,
    )

    fig = page(2, "Controller and cache-state design")
    y = paragraph(
        fig,
        "At each cycle boundary, cached prefix P plus pending token L is the exact committed history. "
        "The target evaluates L and up to k proposed tokens. Upstream returns a matching prefix and a "
        "correction/bonus; Tokenomics commits only through EOG or budget, trims invalid suffixes and "
        "checks cache positions. Retaining valid draft KV avoids replaying rejected suffixes. Full acceptance "
        "can still leave draft state stale; catch-up is charged before re-entry.",
        0.877,
        size=9.5,
    )
    y = paragraph(
        fig,
        "Score(a) = estimated committed tokens / (estimated base milliseconds + full transition cost). "
        "Chosen-action-only EWMA updates use alpha 0.2. Actual positive-token synchronization is removed "
        "from base cost to avoid double counting and is learned per synchronized token. A four-normal-cycle "
        "hold and 5% switch margin limit oscillation; alternate probes rotate every 16 completed cycles. "
        "Request and warm-up state reset to frozen development priors. Calibration restores original target "
        "KV checkpoints and identical prepared draft state; that diagnostic setup is not deployable runtime work.",
        y,
        size=9.5,
    )
    y = paragraph(
        fig,
        f"Across 36 measured requests, ordinary fallback supplies {controller['fallback_tokens']/3456:.1%} "
        f"of 3456 output tokens. There are {controller['probes']} probes and {controller['normal_switches']} "
        "normal incumbent switches. A particular alternate action may wait 80 cycles under an unchanged "
        "incumbent; this is a lightweight experimental estimator, not an optimal scheduler.",
        y,
        size=9.5,
    )
    ax = fig.add_axes([0.055, 0.115, 0.89, 0.47])
    ax.imshow(
        plt.imread(raw / "logging/analysis/development-summary-03-trajectory.png")
    )
    ax.axis("off")
    paragraph(
        fig,
        "Declared worst-regression prompt, measured again with detailed logging. This is a fresh diagnostic "
        "history, not the original minimal timing trace. Probes, fallback, prediction error and acceptance remain visible.",
        0.105,
        size=8.5,
        width=108,
    )

    fig = page(3, "Correctness, overhead and mechanism")
    y = paragraph(
        fig,
        "Both guarded and packed Cloud Release and ASan/UBSan/leak checks pass six suites, including "
        "906 differential fixture cases. Trained controller gates pass 28 records for each drafting mechanism "
        "and exercise all six actions; model drafting observes 40 synchronized tokens. Those focused gates use "
        "explicitly synthetic priors to force behavior, and their timings are excluded from performance claims.",
        0.877,
        size=9.5,
    )
    y = paragraph(
        fig,
        "All 252 validation and 16 separate logging records pass exact token IDs, stops and cache accounting. "
        "The first attempt paused after 25/36 blocks; unchanged resume preserves every earlier raw block, "
        "fit measurement and prior byte-for-byte. All 1,587 payload files plus the final manifest are retained "
        "with ZIP/inner-archive hashes and actual Shutdown evidence.",
        y,
        size=9.5,
    )
    ax = fig.add_axes([0.055, 0.435, 0.89, 0.26])
    ax.imshow(plt.imread(raw / "validation/analysis/controller-comparison.png"))
    ax.axis("off")
    y = heading(fig, "Measured overhead and work", 0.422)
    micro = (costs["decision_ms"] + costs["policy_update_ms"]) / costs["cycles"] * 1000
    y = paragraph(
        fig,
        f"Decision plus update averages {micro:.2f} microseconds per cycle, "
        f"{costs['hot_policy_fraction']:.5%} of generation. Mean initialization is "
        f"{costs['policy_init_ms']/36*1000:.2f} microseconds; parsing is "
        f"{costs['controller_parse_ms']/36*1000:.2f} microseconds per request and remains inside warm latency. "
        f"Mean absolute fractional cycle-time prediction error is {costs['mean_prediction_absolute_fraction']:.1%}.",
        y,
        size=9.5,
    )
    y = paragraph(
        fig,
        "For the same 3456 output tokens, target-only evaluates 3456 target rows, cost-aware 3604 and "
        "cap-8 ngram 4572. Cost-aware avoids 21.2% of cap-8 target work but still evaluates 4.3% more rows "
        "than ordinary decoding. Verification occupies 99.992% of measured generation. Removing all other "
        "time with verification held fixed could improve throughput by only about 0.0083%.",
        y,
        size=9.5,
    )
    y = paragraph(
        fig,
        "No-fallback/cost-aware is 0.9970 [0.9933, 1.0005]; no-probes/cost-aware is 1.0036 "
        "[0.9953, 1.0118]. Detailed/minimal logging is 1.0015 [0.9868, 1.0165] on the separate two-prompt grid. "
        "None shows a practical aggregate gain. Probe execution uses 6.89% of generation, but that is full "
        "probe time including useful committed output, not incremental overhead.",
        y,
        size=9.5,
    )
    paragraph(
        fig,
        "Statistics collapse repetitions before geometric-mean pairing and resample distinct prompts with both caps "
        "together. Repeats are not independent workload samples. All twelve fixed-1 comparison cells are within "
        "the declared 5% practical margin; no cost-aware/ordinary cell is more than 10% slower.",
        y,
        size=9.5,
    )

    fig = page(4, "Reproduce, interpret and extend")
    y = heading(fig, "Native CPU optimization", 0.877)
    if profile is None:
        y = paragraph(
            fig,
            "The verification bottleneck motivates a separate row-consistent packed CPU profile. "
            "Local correctness and sanitizer checks pass, but this report includes no completed trained "
            "packed-profile performance result. A prerequisite gate, paired component comparison and new "
            "controller calibration remain separate from the guarded measurements above.",
            y,
            size=9.5,
        )
    else:
        m = profile["comparisons"]["packed-target/target"]
        warm = profile["warm_latency_speedup_comparisons"]["packed-target/target"]
        packed = profile["baseline_table"]["packed-target"]
        guarded = profile["baseline_table"]["target"]
        y = paragraph(
            fig,
            f"The independently checked 54-record CPU component study measures packed/guarded ordinary "
            f"decode ratio {m['ratio']:.4f} [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}], and warm-latency "
            f"speedup {warm['ratio']:.4f} [{warm['ci95'][0]:.4f}, {warm['ci95'][1]:.4f}]. "
            f"Peak lifetime RSS is {packed['peak_lifetime_rss_bytes']/2**30:.2f} GiB packed "
            f"versus {guarded['peak_lifetime_rss_bytes']/2**30:.2f} GiB guarded. "
            "It uses three development fit prompts, cap128 and three repeats. Both ordinary decoding and "
            "speculation receive the optimization. This is a separate component result, not a controller gain.",
            y,
            size=9.5,
        )
    y = heading(fig, "Build and inspect without model inference", y)
    y = paragraph(
        fig,
        "git clone --recurse-submodules https://github.com/VictorWong123/tokenomics.git\n"
        "cd tokenomics && bash scripts/bootstrap.sh cpu\n"
        ".venv/bin/python scripts/demo_controller.py --max-cycles 12\n"
        ".venv/bin/python -m analysis.engineering_report --out results/local/report\n"
        "# Include a separately verified CPU study with --profile-receipt PATH, if available.",
        y,
        size=8.5,
        width=110,
    )
    y = paragraph(
        fig,
        "The recorded demo verifies the retained trained archive and replays real decisions. "
        "For new model runs, use the documented owned Codespace workflow, exact weights/lock files and "
        "a freshly calibrated operating point. Commands, detailed CLI contracts, analysis and frozen grids "
        "are in README.md, docs/controller-design.md, docs/controller-results.md and docs/evaluation.md.",
        y,
        size=9.5,
    )
    y = heading(fig, "Provenance and ownership", y)
    y = paragraph(
        fig,
        f"Controller inference source: {receipt['source_sha']}. Run {receipt['run_id']}, attempt "
        f"{receipt['attempt']}. llama.cpp pin: fd27b2034905f8dadd8bd6d2cc44595f4929d391. "
        "The canonical receipt records full executable, target, archive and verifier hashes. Raw measurements "
        "remain immutable; tables and this PDF are reproducible derived artifacts.",
        y,
        size=9,
        width=100,
    )
    y = paragraph(
        fig,
        "llama.cpp supplies execution, tokenization, draft generation and common_sampler_sample_and_accept_n. "
        "Tokenomics owns policy/state integration, cache reconciliation, budgets, calibration, tests and "
        "measurement tooling. The packed profile adapts dispatch/layout around existing native CPU kernels; "
        "it does not claim a new GPU kernel. Upstream software and model licenses remain their own; "
        "Qwen2.5-3B has Qwen Research terms. No weights or credentials are committed.",
        y,
        size=9.5,
    )
    y = heading(fig, "Limits and next evidence", y)
    y = paragraph(
        fig,
        "Full 90-prompt held-out model evaluation and Mac/Metal execution remain unexecuted. "
        "The bounded development results do not establish universal acceleration, model quality, energy "
        "or dollar savings, stochastic correctness or research novelty. Prior cost estimates can become stale; "
        "transition charging is conservative over one cycle. A different executable/backend needs new "
        "calibration and its own ordinary baseline.",
        y,
        size=9.5,
    )
    paragraph(
        fig,
        "Primary references and implementation differences are documented in docs/related-work.md and "
        "docs/upstream.md, including Leviathan et al. (2023), Chen et al. (2023), dynamic-lookahead work and "
        "the stock executable's different stopping/counting endpoint. The next optimization follows measured "
        "costs rather than interpreting high acceptance alone as speedup.",
        y,
        size=9.5,
    )

    metadata = dict(
        Title="Tokenomics: C++ runtime engineering checkpoint",
        Author="Tokenomics project",
        Subject="Four-page bounded development engineering report",
        CreationDate=dt.datetime(2026, 11, 8, tzinfo=dt.timezone.utc),
        ModDate=None,
    )
    with PdfPages(output, metadata=metadata) as pdf:
        for i, fig in enumerate(pages, 1):
            pdf.savefig(fig)
            if preview_dir:
                fig.savefig(preview_dir / f"page-{i}.png", dpi=110)
            plt.close(fig)
    atomic_json(
        out / "report-provenance.json",
        dict(
            schema_version=1,
            scope="Four-page bounded development engineering checkpoint, not full held-out research completion.",
            pdf_sha256=sha256(output),
            generator_sha256=sha256(Path(__file__)),
            controller_receipt_sha256=sha256(canonical),
            profile_receipt_sha256=sha256(profile_receipt) if profile_receipt else None,
            profile_receipt_path=str(profile_receipt) if profile_receipt else None,
            figure_sha256={
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    raw / "validation/analysis/controller-comparison.png",
                    raw / "logging/analysis/development-summary-03-trajectory.png",
                )
            },
            pages=4,
            source_inference_sha=receipt["source_sha"],
        ),
    )
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile-receipt", type=Path)
    p.add_argument("--preview-dir", type=Path)
    args = p.parse_args()
    print(generate(args.out, args.profile_receipt, args.preview_dir))
