"""CPU-only coverage for quality-first GGUF baseline discovery."""

import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from calibration.gguf_catalog import (
    gguf_payload_bytes, inventory, preflight_locally_inspected_weights,
    quantization_priority,
)
from calibration.baseline_discovery import (
    all_gpu_rules, choose_baseline, native_confirmation,
)
from model import ModelManager


LAYOUT = {
    "metadata": {"general.architecture": "gemma2", "gemma2.block_count": 2},
    "tensors": [
        {"name": "token_embd.weight", "bytes": 80},
        {"name": "output_norm.weight", "bytes": 12},
        {"name": "blk.0.attn_k.weight", "bytes": 15},
        {"name": "blk.0.ffn_gate.weight", "bytes": 20},
        {"name": "blk.1.attn_k.weight", "bytes": 15},
    ],
    "weights_bytes": 142,
}


def catalog_entry(quant, rank, size=1000):
    return {
        "repo_id": "bartowski/gemma-2-9b-it-GGUF",
        "filename": f"gemma-2-9b-it-{quant}.gguf",
        "quantization": quant,
        "quality_priority": rank,
        "file_bytes": size,
    }


def gpu_status():
    return {
        "ready": True,
        "gpu": {"name": "Mock CUDA", "uuid": "GPU-fake", "index": 0},
        "memory": {"total_mib": 6141.0},
        "errors": [], "foreign_processes": [],
    }


class CatalogTests(unittest.TestCase):
    def test_quant_order_is_publisher_based_not_file_size(self):
        siblings = [
            SimpleNamespace(rfilename="gemma-2-9b-it-Q4_K_M.gguf", size=500),
            SimpleNamespace(rfilename="README.md", size=200),
            SimpleNamespace(rfilename="gemma-2-9b-it-Q6_K.gguf", size=700),
            SimpleNamespace(rfilename="gemma-2-9b-it-Q4_K_L.gguf", size=600),
        ]
        with patch("huggingface_hub.HfApi") as api:
            api.return_value.model_info.return_value.siblings = siblings
            found = inventory()
        self.assertEqual([c["quantization"] for c in found],
                         ["Q6_K", "Q4_K_L", "Q4_K_M"])
        self.assertTrue(all(c["quality_priority"] is not None for c in found))

    def test_unknown_quant_is_visible_but_unranked(self):
        with patch("huggingface_hub.HfApi") as api:
            api.return_value.model_info.return_value.siblings = [
                SimpleNamespace(
                    rfilename="gemma-2-9b-it-FUTURE_QUANT.gguf", size=100
                ),
                SimpleNamespace(
                    rfilename="gemma-2-9b-it-Q4_K_M.gguf", size=500
                ),
            ]
            found = inventory()
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]["quantization"], "Q4_K_M")
        self.assertEqual(found[1]["quality_rank_confidence"], "unranked")
        self.assertIsNone(found[1]["quality_priority"])

    def test_all_nine_publisher_variants_are_ranked(self):
        variants = (
            "IQ2_S", "IQ2_XS", "Q3_K_L-Q8", "Q3_K_XL",
            "Q4_K_M-fp16", "Q6_K-Q8", "Q6_K-f32",
            "Q8_0-f16", "Q8_0_L",
        )
        for label in variants:
            with self.subTest(label=label):
                classification = quantization_priority(label)
                self.assertIsInstance(classification["quality_priority"], int)
                self.assertNotEqual(classification["quality_rank_confidence"], "unranked")

    def test_mixed_precision_variants_rank_before_plain_parent(self):
        for child, parent in (
            ("Q3_K_L-Q8", "Q3_K_L"),
            ("Q4_K_M-fp16", "Q4_K_M"),
            ("Q6_K-Q8", "Q6_K"),
            ("Q6_K-f32", "Q6_K"),
            ("Q8_0-f16", "Q8_0"),
        ):
            with self.subTest(child=child):
                self.assertLess(
                    quantization_priority(child)["quality_priority"],
                    quantization_priority(parent)["quality_priority"],
                )
        self.assertLess(
            quantization_priority("Q8_0_L")["quality_priority"],
            quantization_priority("Q8_0")["quality_priority"],
        )
        self.assertLess(
            quantization_priority("Q3_K_XL")["quality_priority"],
            quantization_priority("Q3_K_L")["quality_priority"],
        )

    def test_live_inventory_all_nine_extra_types_not_blocked(self):
        variant_list = (
            "IQ2_S", "IQ2_XS", "Q3_K_L-Q8", "Q3_K_XL",
            "Q4_K_M-fp16", "Q6_K-Q8", "Q6_K-f32",
            "Q8_0-f16", "Q8_0_L",
        )
        siblings = [
            SimpleNamespace(rfilename=f"gemma-2-9b-it-{name}.gguf", size=900)
            for name in variant_list + ("Q4_K_M",)
        ]
        with patch("huggingface_hub.HfApi") as api:
            api.return_value.model_info.return_value.siblings = siblings
            found = inventory()
        self.assertEqual(len(found), 10)
        self.assertFalse(any(c["quality_priority"] is None for c in found))
        self.assertLess(
            next(i for i, c in enumerate(found) if c["quantization"] == "Q4_K_M-fp16"),
            next(i for i, c in enumerate(found) if c["quantization"] == "Q4_K_M"),
        )

    def test_remote_header_exact_tensor_bytes_not_file_bytes(self):
        name = b"test.weight"
        tensor = (
            struct.pack("<Q", len(name)) + name
            + struct.pack("<I", 2)
            + struct.pack("<QQ", 4, 1)
            + struct.pack("<I", 0) + struct.pack("<Q", 0)
        )
        header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + tensor
        padded = header + bytes((-len(header)) % 32) + bytes(16)
        parsed = gguf_payload_bytes(io.BytesIO(padded), len(padded))
        self.assertEqual(parsed["tensor_count"], 1)
        self.assertEqual(parsed["exact_serialized_tensor_bytes"], 16)
        self.assertEqual(parsed["serialized_payload_bytes_upper_bound"], 16)

    def test_preflight_excludes_only_weights_above_total(self):
        self.assertFalse(
            preflight_locally_inspected_weights(LAYOUT, 150)[
                "certainly_cannot_fit_all_gpu_weights"
            ]
        )
        self.assertTrue(
            preflight_locally_inspected_weights(LAYOUT, 141)[
                "certainly_cannot_fit_all_gpu_weights"
            ]
        )

    def test_gpu_override_rules_cover_each_tensor_including_tied_output(self):
        rules, expected = all_gpu_rules(LAYOUT)
        self.assertEqual(expected["host_tensor_names"], [])
        self.assertEqual(len(expected["gpu_tensor_names"]), 5)
        self.assertEqual(expected["expected_native_log_occurrences"],
                         {"token_embd.weight": 2})
        self.assertTrue(all(r["device"] == "gpu" for r in rules))
        native_logs = "\n".join(
            f"tensor {tensor['name']} (1 MiB Q4) buffer type overridden to CUDA0"
            for tensor in LAYOUT["tensors"]
        ) + ("\ntensor token_embd.weight (1 MiB Q4) "
             "buffer type overridden to CUDA0\n")
        evidence = native_confirmation(LAYOUT, native_logs)
        self.assertTrue(evidence["verified"])
        self.assertTrue(evidence["tied_embedding_output_verified"])
        self.assertEqual(evidence["unverified_names"], [])
        self.assertFalse(native_confirmation(
            LAYOUT, native_logs.replace("CUDA0", "CPU", 1)
        )["verified"])


class BaselineSelectionTests(unittest.TestCase):
    def fake_catalog(self):
        return [catalog_entry("Q5_K_M", 5),
                catalog_entry("Q4_K_L", 7),
                catalog_entry("Q4_K_M", 8)]

    def test_quality_first_exclusion_test_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = []

            def fake_download(repo_id, filename, local_dir):
                downloaded.append(filename)
                file = Path(local_dir) / filename
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"placeholder")
                return str(file)

            def fake_probe(path, **kwargs):
                if "Q4_K_L" in path.name:
                    return {
                        "success": False,
                        "failure_hint": "ambiguous_context_initialization",
                        "error": "ValueError: Failed to create llama_context",
                        "suspected_cuda_oom": False,
                    }
                return {
                    "success": True,
                    "full_gpu_native_verification": {"verified": True},
                    "measurements": {"minimal_inference_completed": True},
                }

            with (
                patch("calibration.baseline_discovery.GPUManager.calibration_status",
                      return_value=gpu_status()),
                patch("calibration.baseline_discovery.inventory",
                      return_value=self.fake_catalog()),
                patch("calibration.baseline_discovery.remote_header",
                      side_effect=lambda repo, name, size: {
                          "exact_serialized_tensor_bytes": (
                              7_000_000_000 if "Q5_K_M" in name else 5_600_000_000
                          ),
                      }),
                patch("calibration.baseline_discovery.inspect_gguf",
                      return_value=LAYOUT),
                patch("calibration.baseline_discovery.probe_gpu_candidate",
                      side_effect=fake_probe),
                patch("huggingface_hub.hf_hub_download", side_effect=fake_download),
            ):
                report = choose_baseline(root=Path(tmp) / "gemma-2-9b-it")
            self.assertTrue(report["success"])
            self.assertTrue(report["next_phase_unlocked"])
            self.assertFalse(report["profile_generated"])
            self.assertEqual(report["baseline"]["quantization"], "Q4_K_M")
            self.assertEqual(downloaded, [
                "gemma-2-9b-it-Q4_K_L.gguf",
                "gemma-2-9b-it-Q4_K_M.gguf",
            ])
            self.assertEqual(
                [t["status"] for t in report["attempts"]],
                ["excluded_weight_payload_exceeds_total_vram",
                 "failed_full_gpu_native_probe", "selected_full_gpu_baseline"]
            )
            self.assertTrue(report["attempts"][0]["future_cpu_gpu_tuning_allowed"])
            path = Path(tmp) / "gemma-2-9b-it" / "baseline_selection.json"
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text())["quantization"], "Q4_K_M")

            selected = ModelManager(root=Path(tmp)).selected_baseline()
            self.assertEqual(selected["quantization"], "Q4_K_M")
            self.assertTrue(selected["native_verified_all_gpu"])
            # Selection is deliberately explicit: get() remains the
            # standard catalog path, and no operational profile is written.
            self.assertEqual(ModelManager(root=Path(tmp)).get(),
                             Path(tmp) / "gemma-2-9b-it" / "gemma-2-9b-it-Q4_K_M.gguf")

    def test_inventory_only_never_downloads_or_writes_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("calibration.baseline_discovery.GPUManager.calibration_status",
                      return_value=gpu_status()),
                patch("calibration.baseline_discovery.inventory",
                      return_value=self.fake_catalog()),
                patch("huggingface_hub.hf_hub_download") as downloader,
            ):
                report = choose_baseline(root=Path(tmp), inventory_only=True)
            self.assertTrue(report["inventory_only"])
            self.assertEqual(len(report["inventory"]), 3)
            self.assertFalse(report["success"])
            self.assertEqual(report["attempts"], [])
            self.assertFalse((Path(tmp) / "baseline_selection.json").exists())
            downloader.assert_not_called()

    def test_success_without_complete_native_verification_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "gemma-2-9b-it-Q4_K_M.gguf"
            file.write_bytes(b"placeholder")
            with (
                patch("calibration.baseline_discovery.GPUManager.calibration_status",
                      return_value=gpu_status()),
                patch("calibration.baseline_discovery.inventory",
                      return_value=[catalog_entry("Q4_K_M", 8)]),
                patch("calibration.baseline_discovery.inspect_gguf",
                      return_value=LAYOUT),
                patch("calibration.baseline_discovery.probe_gpu_candidate",
                      return_value={
                          "success": True,
                          "full_gpu_native_verification": {"verified": False},
                          "measurements": {"minimal_inference_completed": True},
                      }),
            ):
                report = choose_baseline(root=Path(tmp))
            self.assertFalse(report["success"])
            self.assertIsNone(report["baseline"])
            self.assertEqual(len(report["attempts"]), 1)
            self.assertTrue(report["attempts"][0]["needs_diagnostic_review"])
            self.assertFalse((Path(tmp) / "baseline_selection.json").exists())

    def test_unknown_candidate_prevents_selection_without_downloading(self):
        with tempfile.TemporaryDirectory() as tmp:
            unranked = {
                **catalog_entry("UNKNOWN_NEW_SCHEME", None),
                "quality_rank_confidence": "unranked",
            }
            with (
                patch("calibration.baseline_discovery.GPUManager.calibration_status",
                      return_value=gpu_status()),
                patch("calibration.baseline_discovery.inventory",
                      return_value=[catalog_entry("Q4_K_M", 900), unranked]),
                patch("huggingface_hub.hf_hub_download") as downloader,
            ):
                result = choose_baseline(root=Path(tmp))
                inventory_only = choose_baseline(root=Path(tmp), inventory_only=True)
            self.assertFalse(result["success"])
            self.assertEqual(result["unranked_quantizations"], [
                "gemma-2-9b-it-UNKNOWN_NEW_SCHEME.gguf"
            ])
            self.assertIn("Quantizzazioni non classificabili", result["error"])
            self.assertTrue(inventory_only["inventory_only"])
            self.assertEqual(len(inventory_only["inventory"]), 2)
            downloader.assert_not_called()

    def test_unclassified_load_failure_stops_instead_of_skipping_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gemma-2-9b-it-Q4_K_L.gguf"
            path.write_bytes(b"placeholder")
            with (
                patch("calibration.baseline_discovery.GPUManager.calibration_status",
                      return_value=gpu_status()),
                patch("calibration.baseline_discovery.inventory",
                      return_value=self.fake_catalog()[1:]),
                patch("calibration.baseline_discovery.inspect_gguf",
                      return_value=LAYOUT),
                patch("calibration.baseline_discovery.probe_gpu_candidate",
                      return_value={"success": False, "error": "Adapter broken"}),
            ):
                report = choose_baseline(root=Path(tmp))
            self.assertFalse(report["success"])
            self.assertIsNone(report["baseline"])
            self.assertTrue(report["attempts"][0]["needs_diagnostic_review"])
            self.assertEqual(len(report["attempts"]), 1)
            self.assertFalse((Path(tmp) / "baseline_selection.json").exists())


if __name__ == "__main__":
    unittest.main()
