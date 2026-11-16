"""Header-only parser tests; no weight data or network access."""

import struct
import unittest

from scripts.inspect_gguf_header import LIMIT, parse_prefix, projection_census


def text(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def fixture():
    # Includes a discarded vocabulary array so tensor-table positioning is
    # checked independently of retained scalar/string metadata.
    return (
        b"GGUF"
        + struct.pack("<IQQ", 3, 1, 3)
        + text("general.architecture")
        + struct.pack("<I", 8)
        + text("qwen2")
        + text("qwen2.embedding_length")
        + struct.pack("<II", 4, 896)
        + text("tokenizer.ggml.tokens")
        + struct.pack("<IIQ", 9, 8, 2)
        + text("discarded-a")
        + text("discarded-b")
        + text("output.weight")
        + struct.pack("<IQQIQ", 2, 896, 151936, 8, 0)
    )


class GGUFHeaderTests(unittest.TestCase):
    def test_reads_tensor_types_without_weights_or_tokenizer_content(self):
        raw = fixture()
        result = parse_prefix(raw)
        self.assertEqual(result["header_bytes_parsed"], len(raw))
        self.assertEqual(
            result["metadata"],
            {
                "general.architecture": "qwen2",
                "qwen2.embedding_length": 896,
            },
        )
        self.assertEqual(result["tensor_types"], {"Q8_0": 1})
        self.assertEqual(result["tensors"][0]["shape"], [896, 151936])
        self.assertNotIn("discarded", str(result))

    def test_rejects_every_truncated_header(self):
        raw = fixture()
        for size in range(len(raw)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                parse_prefix(raw[:size])

    def test_rejects_bad_magic_version_and_excessive_fields(self):
        for raw in (
            b"HTML" + fixture()[4:],
            b"GGUF" + struct.pack("<IQQ", 99, 0, 0),
            b"GGUF" + struct.pack("<IQQ", 3, 100001, 0),
            b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, LIMIT + 1),
        ):
            with self.assertRaises(ValueError):
                parse_prefix(raw)


class ProjectionCensusTests(unittest.TestCase):
    def test_real_head_sizes_and_embedding_exclusion(self):
        for code, kind, width, expected in (
            (8, "Q8_0", 896, 144643072),
            (14, "Q6_K", 1536, 191439360),
        ):
            rows = [
                {
                    "name": "output.weight",
                    "shape": [width, 151936],
                    "type_code": code,
                    "type": kind,
                },
                {
                    "name": "token_embd.weight",
                    "shape": [256, 10],
                    "type_code": 12,
                    "type": "Q4_K",
                },
                {
                    "name": "blk.0.attn_norm.weight",
                    "shape": [width],
                    "type_code": 0,
                    "type": "F32",
                },
            ]
            result = projection_census(rows)
            self.assertEqual(result["projection_bytes"], expected)
            self.assertEqual(result["embedding_tensor_bytes"], 1440)
            self.assertEqual(result["other_tensor_bytes"], width * 4)
            self.assertEqual(
                result["total_encoded_tensor_bytes"], expected + 1440 + width * 4
            )
            self.assertEqual(result["projection_by_role"]["output"]["tensors"], 1)

    def test_rejects_unknown_or_invalid_tensor_accounting(self):
        row = {
            "name": "blk.0.ffn_down.weight",
            "shape": [8960, 1536],
            "type_code": 14,
            "type": "Q6_K",
        }
        self.assertEqual(projection_census([row])["projection_bytes"], 11289600)
        for rows in (
            [row, row],
            [{**row, "shape": [8959, 1536]}],
            [{**row, "name": "unknown.weight"}],
            [{**row, "type_code": 99}],
        ):
            with self.assertRaises(ValueError):
                projection_census(rows)


if __name__ == "__main__":
    unittest.main()
