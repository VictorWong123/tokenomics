"""Header-only parser tests; no weight data or network access."""

import struct
import unittest

from scripts.inspect_gguf_header import LIMIT, parse_prefix


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


if __name__ == "__main__":
    unittest.main()
