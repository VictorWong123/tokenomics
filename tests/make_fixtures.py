"""Tiny deterministic transformers for state tests, never performance evidence."""

import argparse
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
import gguf


def quantize_k_fixture(value):
    """Simple valid Q4_K blocks for kernel tests, not a production quantizer."""
    blocks = value.reshape(-1, 256)
    low = np.minimum(blocks.min(axis=1, keepdims=True), 0)
    high = np.maximum(blocks.max(axis=1, keepdims=True), 0)
    step = (high - low) / 15
    codes = (
        np.rint((blocks - low) / np.where(step == 0, 1, step))
        .clip(0, 15)
        .astype(np.uint8)
    )
    codes = codes.reshape(-1, 4, 2, 32)
    packed = (codes[:, :, 0] | (codes[:, :, 1] << 4)).reshape(-1, 128)
    # Each packed six-bit scale and minimum is 32.
    scales = np.tile(np.array([160] * 8 + [0] * 4, dtype=np.uint8), (len(blocks), 1))
    raw = np.concatenate(
        [
            (step / 32).astype("<f2").view(np.uint8),
            (-low / 32).astype("<f2").view(np.uint8),
            scales,
            packed,
        ],
        axis=1,
    )
    raw = raw.reshape(value.shape[0], -1)
    restored = gguf.dequantize(raw, gguf.GGMLQuantizationType.Q4_K)
    if (
        not np.isfinite(restored).all()
        or np.max(np.abs(restored - value)) > np.max(step) * 0.55 + 1e-3
    ):
        raise RuntimeError("Invalid Q4_K test fixture encoding")
    return raw


def create(path, winner, seed=31, random=False, quantized=False, k_quantized=False):
    rng = np.random.default_rng(seed)
    width = 256 if quantized or k_quantized else 32
    hidden = 2 * width
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_name("Tokenomics synthetic correctness fixture; NOT a language model")
    w.add_context_length(512)
    w.add_embedding_length(width)
    w.add_block_count(2)
    w.add_feed_forward_length(hidden)
    w.add_head_count(4)
    w.add_head_count_kv(4)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(width // 4)
    w.add_tokenizer_model("llama")
    tokens = (
        ["<unk>", "<s>", "</s>"]
        + [f"<0x{i:02X}>" for i in range(256)]
        + ["▁a", "▁b", "▁"]
    )
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * len(tokens))
    w.add_token_types([2, 3, 3] + [6] * 256 + [1, 1, 1])
    w.add_unk_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_add_bos_token(True)
    w.add_add_eos_token(False)
    n = len(tokens)
    if random:
        emb = rng.normal(0, 0.2, (n, width)).astype(np.float32)
        out = rng.normal(0, 0.2, (n, width)).astype(np.float32)
        if k_quantized:
            out[2] = 0  # Keep the long-prefix regression generating past EOS.
    else:
        emb = np.ones((n, width), np.float32)
        out = np.zeros((n, width), np.float32)
        out[winner] = 1

    def tensor(name, value):
        if k_quantized and value.ndim == 2:
            w.add_tensor(
                name,
                quantize_k_fixture(value),
                raw_dtype=gguf.GGMLQuantizationType.Q4_K,
            )
        elif quantized and value.ndim == 2:
            kind = gguf.GGMLQuantizationType.Q4_0
            w.add_tensor(name, gguf.quantize(value, kind), raw_dtype=kind)
        else:
            w.add_tensor(name, value)

    tensor("token_embd.weight", emb)
    tensor("output.weight", out)
    tensor("output_norm.weight", np.ones(width, np.float32))
    for i in range(2):
        for name in ("attn_norm", "ffn_norm"):
            tensor(f"blk.{i}.{name}.weight", np.ones(width, np.float32))
        for name, shape in {
            "attn_q": (width, width),
            "attn_k": (width, width),
            "attn_v": (width, width),
            "attn_output": (width, width),
            "ffn_gate": (hidden, width),
            "ffn_up": (hidden, width),
            "ffn_down": (width, hidden),
        }.items():
            x = (
                rng.normal(0, 0.08, shape).astype(np.float32)
                if random
                else np.zeros(shape, np.float32)
            )
            tensor(f"blk.{i}.{name}.weight", x)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=pathlib.Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, win in (("target", 259), ("wrong", 260), ("eos", 2)):
        create(a.out / f"{name}.gguf", win)
    create(a.out / "random.gguf", 259, random=True)
    create(a.out / "quantized.gguf", 259, random=True, quantized=True)
    create(a.out / "quantized-k.gguf", 259, random=True, k_quantized=True)
