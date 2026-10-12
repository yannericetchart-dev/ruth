"""Read-only HF header catalogue and tensor relationship tests (no network)."""

import io
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest.mock import patch

from calibration.gguf_catalog import gguf_payload_bytes
from calibration.layer_matrix import (
    analyze_remote_header_layout, remote_huggingface_matrices,
)
from calibration.layer_dependencies import annotate_matrix
from model import ModelManager


class HeaderCatalogTests(unittest.TestCase):
    def test_header_preserves_tensor_shapes_names_groups_and_legacy_preflight(self):
        from gguf import GGMLQuantizationType

        def txt(item):
            payload = item.encode("utf-8")
            return struct.pack("<Q", len(payload)) + payload

        fields = (
            txt("general.architecture") + struct.pack("<I", 8) + txt("qwen2")
            + txt("qwen2.block_count") + struct.pack("<II", 4, 2)
        )
        tensors = b""
        for name in ("token_embd.weight", "blk.0.attn_q.weight",
                     "blk.1.ffn_up.weight", "output.weight"):
            tensors += (
                txt(name)
                + struct.pack("<I", 2)
                + struct.pack("<QQ", 4, 4)
                + struct.pack("<I", int(GGMLQuantizationType.F32))
                + struct.pack("<Q", 0)
            )
        header = (
            b"GGUF" + struct.pack("<IQQ", 3, 4, 2)
            + fields + tensors
        )
        padded = header + b"\x00" * (128 + (-len(header) % 32))
        original = gguf_payload_bytes(io.BytesIO(padded), len(padded))
        enriched = gguf_payload_bytes(
            io.BytesIO(padded), len(padded), include_tensors=True
        )
        self.assertEqual(original["exact_serialized_tensor_bytes"], 64)
        self.assertEqual(enriched["exact_serialized_tensor_bytes"], 64)
        self.assertEqual(enriched["metadata"]["general.architecture"], "qwen2")
        self.assertEqual(enriched["metadata"]["qwen2.block_count"], 2)
        self.assertEqual(len(enriched["tensors"]), 4)
        self.assertEqual(enriched["tensors"][1]["block"], 0)
        self.assertEqual(enriched["tensors"][1]["group"], "attention")
        self.assertEqual(enriched["tensors"][1]["shape"], [4, 4])
        self.assertEqual(enriched["tensors"][1]["bytes"], 64)

    def test_four_hf_reference_matrices_without_weight_download_or_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            called = []

            def fake_inventory(repository):
                found = [
                    s for s in ModelManager.CATALOG.values()
                    if s["repository"] == repository
                ]
                self.assertEqual(len(found), 1)
                spec = found[0]
                return [{
                    "filename": spec["filename"],
                    "file_bytes": 4000,
                    "quantization": spec["quantization"],
                }]

            def fake_header(repo, name, file_bytes, *, include_tensors):
                self.assertTrue(include_tensors)
                self.assertEqual(file_bytes, 4000)
                spec = next(x for x in ModelManager.CATALOG.values()
                            if x["repository"] == repo)
                called.append(repo)
                names = [
                    ("token_embd.weight", 100, None, "embedding"),
                    ("blk.0.attn_q.weight", 100, 0, "attention"),
                    ("blk.1.ffn_up.weight", 100, 1, "feed_forward"),
                ]
                if spec["architecture"] != "gemma2":
                    names.append(("output.weight", 100, None, "output"))
                return {
                    "metadata": {
                        "general.architecture": spec["architecture"],
                        f"{spec['architecture']}.block_count": 2,
                    },
                    "tensor_count": len(names),
                    "weights_bytes": sum(n[1] for n in names),
                    "tensors": [
                        {
                            "name": n, "bytes": b, "block": block,
                            "group": group, "type": "Q4_K_M",
                            "shape": [8, 8],
                        }
                        for n, b, block, group in names
                    ],
                }

            with patch("hardware.GPUManager.calibration_status") as gpu:
                reports = remote_huggingface_matrices(
                    sorted(ModelManager.CATALOG), output_root=output,
                    catalog_inventory=fake_inventory,
                    ranged_header=fake_header,
                )
            gpu.assert_not_called()
            self.assertEqual(len(reports), 4)
            self.assertEqual(len(called), 4)
            for report in reports:
                self.assertEqual(report["schema_version"], 2)
                self.assertFalse(report["baseline_native_verified_all_gpu"])
                self.assertFalse(report["cpu_offload_executed"])
                self.assertEqual(report["block_count"], 2)
                self.assertEqual(
                    report["provenance"],
                    "remote_gguf_header_only_no_runtime_verification",
                )
                target = Path(report["report_path"])
                self.assertTrue(target.is_file())
                persisted = json.loads(target.read_text())
                self.assertEqual(
                    len(persisted["tensor_index"]), persisted["tensor_count"]
                )

    def test_architectural_links_and_tied_gemma_embedding_are_conservative(self):
        source = {
            "architecture": "gemma2", "block_count": 2,
            "layers": [
                {"index": 0, "tensors": [
                    {"name": "blk.0.attn_q.weight", "group": "attention"},
                    {"name": "blk.0.attn_k.weight", "group": "attention"},
                ]},
                {"index": 1, "tensors": [
                    {"name": "blk.1.ffn_up.weight", "group": "feed_forward"},
                ]},
            ],
            "non_block_tensors": [
                {"name": "token_embd.weight", "group": "embedding"},
            ],
        }
        report = annotate_matrix(source)
        self.assertEqual(report["schema_version"], 2)
        by_name = {x["name"]: x for x in report["tensor_index"]}
        self.assertEqual(
            by_name["token_embd.weight"]["placement_constraint"],
            "gemma2_tied_embedding_and_output_must_move_together",
        )
        self.assertEqual(
            by_name["blk.0.attn_q.weight"]["related_tensors_same_subsystem"],
            ["blk.0.attn_k.weight"],
        )
        self.assertEqual(report["layers"][0]["next_block"], 1)
        self.assertFalse(report["relationships"][0]["runtime_execution_graph_verified"])
        self.assertEqual(report["tensor_cpu_native_verified_count"], 0)
        self.assertTrue(report["placement_constraints"])
        for item in report["tensor_index"]:
            self.assertTrue(re.fullmatch(
                item["exact_cpu_override_regex"], item["name"]
            ))

    def test_gguf_reference_is_not_a_cuda_mapping_probe(self):
        layout = {
            "metadata": {"general.architecture": "qwen2",
                         "qwen2.block_count": 1},
            "weights_bytes": 32,
            "tensor_count": 1,
            "tensors": [{
                "name": "blk.0.attn_q.weight", "block": 0,
                "type": "F32", "shape": [8], "bytes": 32,
                "group": "attention",
            }],
        }
        with patch("calibration.layer_matrix.inspect_gguf") as local:
            report = analyze_remote_header_layout(
                layout, model_id="qwen-2.5-7b-instruct",
                repository="bartowski/Qwen2.5-7B-Instruct-GGUF",
                filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                quantization="Q4_K_M",
            )
        local.assert_not_called()
        self.assertFalse(report["baseline_native_verified_all_gpu"])
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["tensor_index"][0]["tensor_role"], "attn_q")


if __name__ == "__main__":
    unittest.main()
