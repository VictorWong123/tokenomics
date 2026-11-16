"""Validate synthetic Metal matrix diagnostics without inferring generation speed."""

import math
import statistics

PROFILES = ("short-mv", "interleaved", "q8-group", "q8-shared")
ROWS = (1, 2, 3, 5, 9)
SHAPES = (
    ("q8_0", 896, 151936),
    ("q5_0", 896, 896),
    ("q4_K", 4864, 896),
    ("q6_K", 4864, 896),
)

PROFILES_BY_MODEL = {"qwen25-05": PROFILES, "qwen25-15": PROFILES[:2]}
SHAPES_BY_MODEL = {
    "qwen25-05": SHAPES,
    "qwen25-15": (
        ("q6_K", 1536, 151936),
        ("q4_K", 1536, 1536),
        ("q6_K", 8960, 1536),
    ),
}
BLOCKS = 4


def encoded_weight_bytes(kind, width, outputs):
    block, size = {
        "q8_0": (32, 34),
        "q5_0": (32, 22),
        "q4_K": (256, 144),
        "q6_K": (256, 210),
    }[kind]
    if width <= 0 or outputs <= 0 or width % block:
        raise ValueError("Invalid quantized matrix geometry")
    return width // block * size * outputs


def summarize(records, model_id="qwen25-05"):
    if model_id not in PROFILES_BY_MODEL:
        raise ValueError("Unsupported component model geometry")
    profiles, shapes = PROFILES_BY_MODEL[model_id], SHAPES_BY_MODEL[model_id]
    gates, timings = {}, {}
    for record in records:
        profile, block = record["profile"], record["block"]
        shape = (record["type"], record["width"], record["outputs"])
        rows = record["rows"]
        if (
            profile not in profiles
            or block not in range(BLOCKS)
            or shape not in shapes
            or rows not in ROWS
            or record.get("model_id", "qwen25-05") != model_id
            or record["synthetic_fixture"]
            or not record["backend"].startswith("MTL")
        ):
            raise ValueError("Unexpected component profile, shape or backend")
        key = (profile, block, *shape, rows)
        if record["state"] == "bitwise_passed":
            if (
                key in gates
                or record["compared_values"] != shape[2] * rows
                or record["weight_bytes"] != encoded_weight_bytes(*shape)
            ):
                raise ValueError("Duplicate or incomplete component arithmetic gate")
            gates[key] = record
        elif record["state"] == "timing":
            trial = record["trial"]
            if trial not in range(16) or record["repetitions"] != 8:
                raise ValueError("Unexpected component trial schedule")
            for field in ("batch_us", "serial_graph_us"):
                value = record[field]
                if (
                    not isinstance(value, (float, int))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("Invalid component duration")
            trial_key = (*key, trial)
            if trial_key in timings:
                raise ValueError("Duplicate component timing")
            timings[trial_key] = record
        else:
            raise ValueError("Failed or unknown component record")
    expected = {
        (profile, block, *shape, rows)
        for profile in profiles
        for block in range(BLOCKS)
        for shape in shapes
        for rows in ROWS
    }
    if set(gates) != expected or set(timings) != {
        (*key, trial) for key in expected for trial in range(16)
    }:
        raise ValueError("Missing component gates or timings")
    identities = {}
    for key, gate in gates.items():
        cell = key[2:]
        identity = tuple(
            gate[field]
            for field in (
                "weight_bytes",
                "weight_fnv1a64",
                "input_fnv1a64",
                "output_fnv1a64",
            )
        )
        if identities.setdefault(cell, identity) != identity:
            raise ValueError(
                "Matrix inputs or exact outputs changed across profiles/blocks"
            )
    for key, timing in timings.items():
        gate = gates[key[:-1]]
        if any(
            timing[field] != gate[field] for field in ("weight_bytes", "weight_fnv1a64")
        ):
            raise ValueError("Timed weights differ from their arithmetic gate")
    cells = []
    for shape in shapes:
        for rows in ROWS:
            values = {}
            for profile in profiles:
                blocks = []
                for block in range(BLOCKS):
                    trials = [
                        timings[(profile, block, *shape, rows, trial)]
                        for trial in range(16)
                    ]
                    blocks.append(
                        {
                            "block": block,
                            "median_batch_us": statistics.median(
                                t["batch_us"] for t in trials
                            ),
                            "median_serial_graph_us": statistics.median(
                                t["serial_graph_us"] for t in trials
                            ),
                            "median_serial_graph_ratio": statistics.median(
                                t["serial_graph_us"] / t["batch_us"] for t in trials
                            ),
                        }
                    )
                values[profile] = {"blocks": blocks}
            for profile in profiles:
                ratios = [
                    baseline["median_batch_us"] / candidate["median_batch_us"]
                    for baseline, candidate in zip(
                        values["short-mv"]["blocks"], values[profile]["blocks"]
                    )
                ]
                values[profile]["ratio_vs_short_mv_by_block"] = ratios
                values[profile]["median_ratio_vs_short_mv"] = statistics.median(ratios)
            cells.append(
                {
                    "type": shape[0],
                    "width": shape[1],
                    "outputs": shape[2],
                    "rows": rows,
                    "profiles": values,
                }
            )
    return {
        "state": "passed",
        "model_id": model_id,
        "profiles": list(profiles),
        "blocks": BLOCKS,
        "scope": "Synthetic completed-work matrix component only; no trained-token or generation speed claim",
        "compared_values_per_profile_block": sum(
            s[2] * r for s in shapes for r in ROWS
        ),
        "trial_pairs": len(timings),
        "limitations": [
            "Synthetic weights/activations; shapes borrowed from the pinned target, not its learned weight values",
            "Serial control is a single graph containing one-row matrix operations; it excludes per-token model/context overhead",
            "Repeated resident inputs are cache-warm; the output head exceeds cache but this does not reproduce full-model cache traffic",
            "Four rotated profile blocks describe variation; their medians are not a confidence interval",
            "FNV checks detect accidental data changes; source/executable identities use SHA256 in the runner manifest",
            "Actual trained cross-profile gates remain required before end-to-end performance measurements",
        ],
        "cells": cells,
    }
