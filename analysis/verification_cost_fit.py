"""Reproduce the diagnostic fixed-cycle/per-row fit; no speed forecast."""

import argparse
from pathlib import Path
import tempfile
import numpy as np

from analysis.proposal_replay import CPU_FIT, MAC_ARCHIVE, extract_mac_fit, load_fit
from bench.common import atomic_json, sha256, source_fingerprint


def fit(directory):
    rows, targets, identity = load_fit(directory)
    result = []
    for target in sorted(targets, key=lambda row: row["prompt_id"]):
        selected = [row for row in rows if row["prompt_id"] == target["prompt_id"]]
        design = np.asarray(
            [
                [row["execution"]["cycles"], row["execution"]["verified_rows"]]
                for row in selected
            ],
            dtype=float,
        )
        observed = np.asarray([row["execution"]["verify_ms"] for row in selected])
        coefficients, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
        residual = design @ coefficients - observed
        result.append(
            {
                "prompt_id": target["prompt_id"],
                "observations": len(selected),
                "rank": int(rank),
                "fixed_cycle_ms": float(coefficients[0]),
                "per_row_ms": float(coefficients[1]),
                "relative_rmse": float(
                    np.sqrt(np.mean(residual**2)) / np.mean(observed)
                ),
                "nonnegative_coefficients": bool(np.all(coefficients >= 0)),
                "observed": [
                    {
                        "configuration_id": row["configuration_id"],
                        "cycles": int(design[i, 0]),
                        "verified_rows": int(design[i, 1]),
                        "verify_ms": float(observed[i]),
                        "residual_ms": float(residual[i]),
                    }
                    for i, row in enumerate(selected)
                ],
            }
        )
    return {"identity": identity, "fits": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite existing report")
    with tempfile.TemporaryDirectory() as tmp:
        mac = fit(extract_mac_fit(Path(tmp)))
    result = {
        "schema_version": 1,
        "source": source_fingerprint(),
        "mac_archive_sha256": sha256(MAC_ARCHIVE),
        "m3_metal_05b": mac,
        "cpu_3b": fit(CPU_FIT),
        "model": "verify_ms = H * cycles + R * verified_rows; unconstrained least squares, separately per fit prompt",
        "limitation": "Six conditions per prompt, one timing observation per condition. Noise, context, graph shapes and paths are not isolated. Negative coefficients and residuals expose model limitations; coefficients do not identify hardware costs or forecast new-policy speed.",
        "speed_claim": None,
    }
    atomic_json(args.out, result)


if __name__ == "__main__":
    main()
