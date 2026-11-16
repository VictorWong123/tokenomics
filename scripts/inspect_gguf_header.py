"""Inspect a pinned public GGUF prefix without saving or redistributing weights."""

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import re
from pathlib import Path
import struct
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 16 * 1024 * 1024
SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
# Pinned ggml type numbers; retain numeric codes for any unlisted future type.
TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    30: "BF16",
}


def parse_prefix(data):
    stream = io.BytesIO(data)

    def take(size):
        if size < 0 or size > LIMIT:
            raise ValueError("Invalid GGUF field size")
        value = stream.read(size)
        if len(value) != size:
            raise ValueError("GGUF header exceeds available prefix")
        return value

    def scalar(code):
        fmt = "<" + SCALARS[code]
        return struct.unpack(fmt, take(struct.calcsize(fmt)))[0]

    def string(keep=True):
        size = scalar(10)
        value = take(size)
        return value.decode("utf-8") if keep else None

    def value(kind, keep=True, depth=0):
        if depth > 2:
            raise ValueError("Nested GGUF array exceeds inspection limit")
        if kind in SCALARS:
            result = scalar(kind)
            return result if keep else None
        if kind == 8:
            return string(keep)
        if kind == 9:
            element, count = scalar(4), scalar(10)
            if count > LIMIT:
                raise ValueError("Excessive GGUF array length")
            # Tokenizer strings can occupy most of the header; do not retain
            # vocabulary/merges or arbitrary model-card strings in the report.
            for _ in range(count):
                value(element, False, depth + 1)
            return {"array_type": element, "count": count} if keep else None
        raise ValueError(f"Unsupported GGUF metadata type {kind}")

    if take(4) != b"GGUF":
        raise ValueError("Not a GGUF file")
    version = scalar(4)
    if version not in (2, 3):
        raise ValueError("Only little-endian GGUF versions2/3 are supported")
    tensors, fields = scalar(10), scalar(10)
    if tensors > 100000 or fields > 100000:
        raise ValueError("Excessive GGUF header counts")
    metadata = {}
    for _ in range(fields):
        key, kind = string(), scalar(4)
        keep = key.startswith("qwen2.") or key in (
            "general.architecture",
            "general.file_type",
            "general.quantization_version",
        )
        result = value(kind, keep)
        if keep:
            metadata[key] = result
    records = []
    for _ in range(tensors):
        name, dimensions = string(), scalar(4)
        if dimensions < 1 or dimensions > 4:
            raise ValueError("Unsupported GGUF tensor dimensionality")
        shape = [scalar(10) for _ in range(dimensions)]
        if any(size <= 0 for size in shape):
            raise ValueError("Non-positive GGUF tensor extent")
        kind, offset = scalar(4), scalar(10)
        records.append(
            {
                "name": name,
                "shape": shape,
                "type_code": kind,
                "type": TYPES.get(kind, f"TYPE_{kind}"),
                "relative_data_offset": offset,
            }
        )
    return {
        "version": version,
        "header_bytes_parsed": stream.tell(),
        "metadata": metadata,
        "tensor_types": dict(Counter(r["type"] for r in records)),
        "tensors": records,
    }


def projection_census(records):
    """Encoded payload accounting, not measured memory traffic or latency."""
    sizes = {
        0: (1, 4),
        1: (1, 2),
        6: (32, 22),
        8: (32, 34),
        12: (256, 144),
        14: (256, 210),
        30: (1, 2),
    }
    total = embedding = other = 0
    by_type, by_role, variants = {}, {}, {}
    names = set()
    for row in records:
        name, shape = row["name"], row["shape"]
        if name in names or row["type_code"] not in sizes:
            raise ValueError("Duplicate tensor or unsupported census quantization")
        names.add(name)
        block, size = sizes[row["type_code"]]
        if not shape or any(value <= 0 for value in shape) or shape[0] % block:
            raise ValueError("Invalid census tensor geometry")
        encoded = math.prod(shape) // block * size
        total += encoded
        if name == "token_embd.weight":
            embedding += encoded
            continue
        if len(shape) != 2:
            other += encoded
            continue
        match = re.fullmatch(
            r"blk\.[0-9]+\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight",
            name,
        )
        if name != "output.weight" and match is None:
            raise ValueError("Unclassified dense-model matrix")
        role = "output" if name == "output.weight" else match[1]
        variant = variants.setdefault(
            (role, row["type"], tuple(shape)), {"tensors": 0, "encoded_bytes": 0}
        )
        variant["tensors"] += 1
        variant["encoded_bytes"] += encoded
        for table, key in ((by_type, row["type"]), (by_role, role)):
            cell = table.setdefault(key, {"tensors": 0, "encoded_bytes": 0})
            cell["tensors"] += 1
            cell["encoded_bytes"] += encoded
    for role, cell in by_role.items():
        cell["variants"] = [
            {"type": kind, "shape": list(shape), **counts}
            for (name, kind, shape), counts in sorted(variants.items())
            if name == role
        ]
    projected = sum(row["encoded_bytes"] for row in by_type.values())
    if projected + embedding + other != total or projected != sum(
        row["encoded_bytes"] for row in by_role.values()
    ):
        raise ValueError("Projection census does not reconcile")
    return {
        "scope": "Logical encoded dense projection payloads; excludes embedding lookups, vector operations, activations, KV, repeated loads and cache effects. Not measured traffic, bandwidth or speed.",
        "total_encoded_tensor_bytes": total,
        "embedding_tensor_bytes": embedding,
        "other_tensor_bytes": other,
        "projection_bytes": projected,
        "projection_by_type": dict(sorted(by_type.items())),
        "projection_by_role": dict(sorted(by_role.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("qwen25-05", "qwen25-15"), default="qwen25-05"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output must be fresh")
    entry = json.loads((ROOT / "configs/metal-model-header-pins-v1.json").read_text())[
        "models"
    ][args.model]
    url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{entry['file']}"
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{LIMIT-1}"})
    report = {
        "model": args.model,
        "source_url": url,
        "lock": entry,
        "full_model_sha256_verified": False,
        "weights_saved": False,
    }
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.status not in (200, 206):
                raise ValueError("Unexpected prefix response status")
            if response.status == 206:
                span = re.fullmatch(
                    r"bytes 0-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                )
                if not span or int(span[2]) != entry["bytes"]:
                    raise ValueError("Pinned GGUF content range/size mismatch")
            elif int(response.headers.get("Content-Length", -1)) != entry["bytes"]:
                raise ValueError("Pinned GGUF content length mismatch")
            prefix = response.read(LIMIT)
            report["http_status"] = response.status
            report["content_range"] = response.headers.get("Content-Range")
        report.update(parse_prefix(prefix))
        report["projection_census"] = projection_census(report["tensors"])
        report["prefix_sha256"] = hashlib.sha256(prefix).hexdigest()
        report["prefix_bytes_read"] = len(prefix)
        report["state"] = "header_inspected"
    except Exception as exc:
        # Do not print redirect URLs, temporary signed parameters or credentials.
        report["state"] = "failed"
        report["error_type"] = type(exc).__name__
        if isinstance(exc, urllib.error.HTTPError):
            report["http_error_code"] = exc.code
        raise RuntimeError("Pinned GGUF prefix inspection failed; see report") from None
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        summary = {
            key: val
            for key, val in report.items()
            if key not in ("tensors", "source_url")
        }
        summary["projection_examples"] = [
            r
            for r in report.get("tensors", [])
            if r["name"]
            in (
                "output.weight",
                "token_embd.weight",
                "blk.0.attn_q.weight",
                "blk.0.ffn_down.weight",
                "blk.0.ffn_up.weight",
                "blk.0.ffn_gate.weight",
            )
        ]
        print("GGUF_HEADER_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
