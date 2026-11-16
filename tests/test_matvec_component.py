import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from analysis.matvec_component import (
    BLOCKS,
    PROFILES_BY_MODEL,
    ROWS,
    SHAPES_BY_MODEL,
    summarize,
    encoded_weight_bytes,
)
from bench.common import ROOT
from scripts import metal_matvec_study


def fixture(model_id="qwen25-05"):
    records = []
    for profile in PROFILES_BY_MODEL[model_id]:
        for block in range(BLOCKS):
            for kind, width, outputs in SHAPES_BY_MODEL[model_id]:
                for rows in ROWS:
                    base = dict(
                        profile=profile,
                        model_id=model_id,
                        block=block,
                        type=kind,
                        width=width,
                        outputs=outputs,
                        rows=rows,
                        backend="MTL0",
                        synthetic_fixture=False,
                        weight_bytes=encoded_weight_bytes(kind, width, outputs),
                        weight_fnv1a64=1,
                    )
                    records.append(
                        dict(
                            base,
                            state="bitwise_passed",
                            compared_values=outputs * rows,
                            input_fnv1a64=rows,
                            output_fnv1a64=rows + 10,
                        )
                    )
                    for trial in range(16):
                        records.append(
                            dict(
                                base,
                                state="timing",
                                trial=trial,
                                repetitions=8,
                                batch_us=10 if profile == "short-mv" else 5,
                                serial_graph_us=20,
                            )
                        )
    return records


class MatvecComponentTests(unittest.TestCase):
    def test_component_ratios_are_separate_from_serial_control(self):
        result = summarize(fixture())
        self.assertEqual(result["trial_pairs"], 5120)
        self.assertEqual(len(result["cells"]), 20)
        values = result["cells"][0]["profiles"]
        self.assertEqual(values["interleaved"]["median_ratio_vs_short_mv"], 2)
        self.assertEqual(
            values["interleaved"]["blocks"][0]["median_serial_graph_ratio"], 4
        )

    def test_missing_duplicate_failure_and_data_changes_rejected(self):
        rows = fixture()
        bad = [rows[:-1], rows + [rows[0]]]
        changed_timing = copy.deepcopy(rows)
        changed_timing[1]["weight_fnv1a64"] = 999
        bad.append(changed_timing)
        for field, value in (
            ("output_fnv1a64", 999),
            ("input_fnv1a64", 999),
            ("weight_fnv1a64", 999),
            ("weight_bytes", 100),
            ("state", "bitwise_mismatch"),
            ("backend", "CPU"),
            ("synthetic_fixture", True),
        ):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            bad.append(changed)
        for changed in bad:
            with self.assertRaises(ValueError):
                summarize(changed)

    def test_invalid_timings_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            rows = fixture()
            rows[1]["batch_us"] = value
            with self.assertRaises(ValueError):
                summarize(rows)


class MatvecModelGeometryTests(unittest.TestCase):
    def test_15_grid_and_exact_header_shapes(self):
        report = summarize(fixture("qwen25-15"), "qwen25-15")
        self.assertEqual(report["trial_pairs"], 1920)
        self.assertEqual(len(report["cells"]), 15)
        self.assertEqual(report["profiles"], ["short-mv", "interleaved"])
        header = json.loads(
            (
                ROOT / "results/verification/model-header-15-20261116/header.json"
            ).read_text()
        )
        examples = {row["name"]: row for row in header["projection_examples"]}
        expected = [
            examples[name]
            for name in (
                "output.weight",
                "blk.0.attn_q.weight",
                "blk.0.ffn_down.weight",
            )
        ]
        self.assertEqual(
            list(SHAPES_BY_MODEL["qwen25-15"]),
            [
                (row["type"].lower().replace("_k", "_K"), *row["shape"])
                for row in expected
            ],
        )
        with self.assertRaises(ValueError):
            summarize(fixture("qwen25-15"))
        with self.assertRaises(ValueError):
            summarize(fixture(), "qwen25-15")

    def test_15_runner_excludes_q8_and_checks_native_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests/test_matvec_component.cpp").write_text("fixture")
            records = fixture("qwen25-15")
            commands = []
            wrong_shapes = False

            def command(args, log):
                args = [str(arg) for arg in args]
                commands.append(args)
                log = Path(log)
                if "--build" in args:
                    build = Path(args[args.index("--build") + 1])
                    build.mkdir(exist_ok=True)
                    (build / "test_matvec_component").write_text("fixture executable")
                if "--shapes" in args:
                    rows = [
                        dict(
                            model_id="qwen25-15",
                            type=kind,
                            width=width,
                            outputs=outputs,
                            weight_bytes=encoded_weight_bytes(kind, width, outputs),
                        )
                        for kind, width, outputs in SHAPES_BY_MODEL["qwen25-15"]
                    ]
                    if wrong_shapes:
                        rows[0]["width"] = 896
                    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
                elif "--metal" in args:
                    profile = Path(args[0]).parent.name.removeprefix("build-metal-")
                    rows = [
                        row
                        for row in records
                        if row["profile"] == profile and row["block"] == 0
                    ]
                    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
                else:
                    log.write_text("fixture build log")

            with patch.object(metal_matvec_study, "ROOT", root), patch.object(
                metal_matvec_study.platform, "system", return_value="Darwin"
            ), patch.object(
                metal_matvec_study.platform, "machine", return_value="arm64"
            ), patch.object(
                metal_matvec_study, "source_fingerprint", return_value={"fixture": True}
            ), patch.object(
                metal_matvec_study, "hardware", return_value={"fixture": True}
            ), patch.object(
                metal_matvec_study,
                "build_configuration",
                return_value={"fixture": True},
            ), patch.object(
                metal_matvec_study, "tool", return_value="cmake"
            ), patch.object(
                metal_matvec_study, "command", side_effect=command
            ):
                metal_matvec_study.study(
                    SimpleNamespace(model_id="qwen25-15", out=root / "out")
                )
                wrong_shapes = True
                with self.assertRaisesRegex(ValueError, "geometry"):
                    metal_matvec_study.study(
                        SimpleNamespace(model_id="qwen25-15", out=root / "wrong-shapes")
                    )
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(manifest["state"], "passed")
            self.assertFalse(manifest["model_weights_loaded"])
            self.assertEqual(set(manifest["binaries"]), {"short-mv", "interleaved"})
            self.assertEqual(sum("--metal" in cmd for cmd in commands), 8)
            self.assertTrue(
                all(
                    "q8-group" not in " ".join(cmd) and "q8-shared" not in " ".join(cmd)
                    for cmd in commands
                )
            )
            for cmd in commands:
                if "-S" in cmd:
                    self.assertIn("-DTOKENOMICS_LOGIT_COPY=OFF", cmd)
                    self.assertIn("-DTOKENOMICS_METAL_Q8_SHARED=OFF", cmd)

    @unittest.skipUnless(
        os.getenv("TOKENOMICS_MATVEC_BINARY"), "requires native matrix component"
    )
    def test_native_geometry_and_small_15_fixture(self):
        binary = os.environ["TOKENOMICS_MATVEC_BINARY"]
        for model_id in SHAPES_BY_MODEL:
            result = subprocess.run(
                [binary, "--shapes", "--model-id", model_id],
                check=True,
                capture_output=True,
                text=True,
            )
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(
                [(row["type"], row["width"], row["outputs"]) for row in rows],
                list(SHAPES_BY_MODEL[model_id]),
            )
            self.assertEqual(
                [row["weight_bytes"] for row in rows],
                [encoded_weight_bytes(*shape) for shape in SHAPES_BY_MODEL[model_id]],
            )
        result = subprocess.run(
            [binary, "--cpu-fixture", "--model-id", "qwen25-15"],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith("{")
        ]
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(row["compared_values"] for row in rows), 5120)
        self.assertTrue(
            all(
                row["state"] == "bitwise_passed"
                and row["synthetic_fixture"]
                and row["model_id"] == "qwen25-15"
                for row in rows
            )
        )
        result = subprocess.run(
            [binary, "--shapes", "--model-id", "unknown"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
