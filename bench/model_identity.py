"""Pinned GGUF identities and header-confirmed geometry for local Metal studies."""

import json

from bench.common import ROOT

DEFAULT_MODEL = "qwen25-05"
MODEL_PINS = json.loads((ROOT / "configs/metal-model-header-pins-v1.json").read_text())[
    "models"
]
# Geometry comes from the retained bounded header inspections, not filenames.
MODEL_GEOMETRY = {
    "qwen25-05": {
        "layers": 24,
        "embedding": 896,
        "ffn": 4864,
        "query_heads": 14,
        "kv_heads": 2,
        "output_type": "Q8_0",
        "q8_group_applicable": True,
    },
    "qwen25-15": {
        "layers": 28,
        "embedding": 1536,
        "ffn": 8960,
        "query_heads": 12,
        "kv_heads": 2,
        "output_type": "Q6_K",
        "q8_group_applicable": False,
    },
}


def model_identity(model_id=DEFAULT_MODEL):
    if model_id not in MODEL_GEOMETRY or model_id not in MODEL_PINS:
        raise ValueError(f"Unsupported pinned model: {model_id}")
    return {"id": model_id, **MODEL_PINS[model_id], **MODEL_GEOMETRY[model_id]}


def require_model_variant(model_id, *, grouped_q8=False):
    identity = model_identity(model_id)
    if grouped_q8 and not identity["q8_group_applicable"]:
        raise ValueError(
            f"{model_id} has no Q8 weights: use plain short-MV, interleaving or direct host sampling"
        )
    return identity
