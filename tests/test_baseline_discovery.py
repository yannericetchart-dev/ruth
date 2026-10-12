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
    all_gpu_rules, choose_baseline, native_confirmation, minimal_chat_prompt,
)
from model import ModelManager
from calibration.mapping_candidates import propose_mapping_candidates


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
                    "architecture": "gemma2",
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


LLAMA_LAYOUT = {
    "metadata": {
        "general.architecture": "llama",
        "llama.block_count": 2,
    },
    "tensors": [
        {"name": "token_embd.weight", "bytes": 60},
        {"name": "output.weight", "bytes": 60},
        {"name": "output_norm.weight", "bytes": 10},
        {"name": "blk.0.attn_k.weight", "bytes": 20},
        {"name": "blk.1.ffn_gate.weight", "bytes": 20},
    ],
    "weights_bytes": 170,
}


class Llama31CompatibilityTests(unittest.TestCase):
    def test_llama_publisher_prefix_and_arm_variants_are_catalogued(self):
        names = [
            "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
            "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "Meta-Llama-3.1-8B-Instruct-IQ4_NL.gguf",
            "Meta-Llama-3.1-8B-Instruct-Q4_0_4_4.gguf",
            "Meta-Llama-3.1-8B-Instruct-Q4_0_4_8.gguf",
            "Meta-Llama-3.1-8B-Instruct-Q4_0_8_8.gguf",
        ]
        with patch("huggingface_hub.HfApi") as api:
            api.return_value.model_info.return_value.siblings = [
                SimpleNamespace(rfilename=name, size=1000) for name in names
            ]
            models = inventory("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
        self.assertEqual(len(models), len(names))
        self.assertEqual(models[0]["quantization"], "Q6_K")
        self.assertFalse(any(m["quality_priority"] is None for m in models))
        special = [
            m for m in models if m["quantization"].startswith("Q4_0_")
        ]
        self.assertEqual(len(special), 3)
        self.assertTrue(all(m["cuda_baseline_eligible"] is False for m in special))

    def test_full_gpu_llama_embedding_and_output_are_independent(self):
        rules, expected = all_gpu_rules(LLAMA_LAYOUT)
        self.assertEqual(expected["expected_native_log_occurrences"], {})
        self.assertEqual(expected["output_weight_target"], "gpu")
        self.assertEqual(set(expected["gpu_tensor_names"]), {
            t["name"] for t in LLAMA_LAYOUT["tensors"]
        })
        self.assertTrue(any("output" in r["pattern"] for r in rules))
        log = "\n".join(
            f"tensor {tensor['name']} (1 MiB Q4) buffer type overridden to CUDA0"
            for tensor in LLAMA_LAYOUT["tensors"]
        )
        check = native_confirmation(LLAMA_LAYOUT, log)
        self.assertTrue(check["verified"])
        self.assertIsNone(check["tied_embedding_output_verified"])
        wrong = log.replace(
            "tensor output.weight (1 MiB Q4) buffer type overridden to CUDA0",
            "tensor output.weight (1 MiB Q4) buffer type overridden to CPU",
        )
        self.assertFalse(native_confirmation(LLAMA_LAYOUT, wrong)["verified"])

    def test_llama_rope_freqs_repeated_per_block_are_all_verified(self):
        layout = {
            "metadata": {
                "general.architecture": "llama",
                "llama.block_count": 32,
            },
            "tensors": LLAMA_LAYOUT["tensors"] + [
                {"name": "rope_freqs.weight", "bytes": 128},
            ],
        }
        _, expected = all_gpu_rules(layout)
        self.assertEqual(
            expected["expected_native_log_occurrences"],
            {"rope_freqs.weight": 32},
        )

        # One serialized rope tensor is loaded by llama.cpp into a native
        # RoPE buffer for each block, as observed in the actual RTX report.
        base = "\n".join(
            f"tensor {item['name']} (1 MiB f32) "
            "buffer type overridden to CUDA0"
            for item in layout["tensors"]
            if item["name"] != "rope_freqs.weight"
        )
        rope = (
            "tensor rope_freqs.weight (0 MiB f32) "
            "buffer type overridden to CUDA0"
        )
        logs = base + "\n" + "\n".join([rope] * 32)
        result = native_confirmation(layout, logs)
        self.assertTrue(result["verified"])
        self.assertIsNone(result["tied_embedding_output_verified"])
        self.assertEqual(result["unverified_names"], [])
        native = result["native_override"]
        self.assertTrue(native["native_logged_names_exact"])
        self.assertEqual(
            native["output_placement_source"],
            "independent_output_native_override_logs",
        )
        self.assertEqual(native["native_logged_tensor_count"], 6)
        self.assertEqual(native["native_logged_occurrence_count"], 37)
        actual = next(
            item for item in native["tensors"]
            if item["name"] == "rope_freqs.weight"
        )
        self.assertEqual(actual["native_occurrences"], 32)
        self.assertTrue(actual["verified_by_log"])

    def test_llama_rope_requires_exact_repetitions_and_gpu_for_all(self):
        layout = {
            "metadata": {
                "general.architecture": "llama",
                "llama.block_count": 3,
            },
            "tensors": LLAMA_LAYOUT["tensors"] + [
                {"name": "rope_freqs.weight", "bytes": 12},
            ],
        }
        base = "\n".join(
            f"tensor {item['name']} (1 MiB f32) "
            "buffer type overridden to CUDA0"
            for item in layout["tensors"]
            if item["name"] != "rope_freqs.weight"
        )
        rope = (
            "tensor rope_freqs.weight (0 MiB f32) "
            "buffer type overridden to CUDA0"
        )
        self.assertTrue(native_confirmation(
            layout, base + "\n" + "\n".join([rope] * 3)
        )["verified"])
        for entries in (
            [rope] * 2,
            [rope] * 4,
            [rope, rope.replace("CUDA0", "CPU"), rope],
            [rope, rope.replace("CUDA0", "CUDA_Host"), rope],
        ):
            with self.subTest(entries=entries):
                self.assertFalse(native_confirmation(
                    layout, base + "\n" + "\n".join(entries)
                )["verified"])

    def test_llama_rope_without_valid_block_count_is_inconclusive(self):
        layout = {
            "metadata": {
                "general.architecture": "llama",
                "llama.block_count": 0,
            },
            "tensors": LLAMA_LAYOUT["tensors"] + [
                {"name": "rope_freqs.weight", "bytes": 12},
            ],
        }
        with self.assertRaises(ValueError):
            all_gpu_rules(layout)

    def test_each_model_uses_its_own_minimal_chat_template(self):
        gemma = minimal_chat_prompt("gemma2")
        llama = minimal_chat_prompt("llama")
        self.assertIn("<start_of_turn>user", gemma)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", llama)
        self.assertIn("<|eot_id|>", llama)
        self.assertNotIn("<|begin_of_text|>", llama)
        self.assertIn("\n\n", llama)
        with self.assertRaises(ValueError):
            minimal_chat_prompt("unknown")

    def test_llama_inventory_mode_does_not_write_gemma_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = [{
                **catalog_entry("Q5_K_M", 700),
                "filename": "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
            }]
            with (
                patch("calibration.baseline_discovery.GPUManager.calibration_status",
                      return_value=gpu_status()),
                patch("calibration.baseline_discovery.inventory",
                      return_value=models),
            ):
                output = choose_baseline(
                    root=Path(tmp) / "llama-3.1-8b-instruct",
                    model_id="llama-3.1-8b-instruct",
                    inventory_only=True,
                )
            self.assertEqual(
                output["repository"],
                "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            )
            self.assertEqual(output["model_id"], "llama-3.1-8b-instruct")
            self.assertEqual(output["expected_architecture"], "llama")
            self.assertEqual(len(output["inventory"]), 1)
            self.assertFalse((Path(tmp) / "gemma-2-9b-it").exists())

    def test_llama_full_gpu_selection_is_separate_and_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "llama-3.1-8b-instruct"
            model_dir.mkdir(parents=True)
            file = model_dir / "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
            file.write_bytes(b"placeholder")
            models = [{
                **catalog_entry("Q5_K_M", 700),
                "filename": file.name,
            }]
            probe = {
                "success": True,
                "architecture": "llama",
                "full_gpu_native_verification": {"verified": True},
                "measurements": {"minimal_inference_completed": True},
            }
            with (
                patch("calibration.baseline_discovery.GPUManager.calibration_status",
                      return_value=gpu_status()),
                patch("calibration.baseline_discovery.inventory", return_value=models),
                patch("calibration.baseline_discovery.inspect_gguf",
                      return_value=LLAMA_LAYOUT),
                patch("calibration.baseline_discovery.probe_gpu_candidate",
                      return_value=probe),
            ):
                report = choose_baseline(root=model_dir,
                                         model_id="llama-3.1-8b-instruct")
            self.assertTrue(report["success"])
            self.assertEqual(report["baseline"]["quantization"], "Q5_K_M")
            self.assertEqual(report["baseline"]["architecture"], "llama")
            selected = ModelManager(root=Path(tmp)).selected_baseline(
                "llama-3.1-8b-instruct"
            )
            self.assertEqual(selected["filename"], file.name)
            self.assertIsNone(ModelManager(root=Path(tmp)).selected_baseline(
                "gemma-2-9b-it"
            ))

    def test_llama_mapping_generator_does_not_tie_embedding_and_output(self):
        layout = {
            "metadata": LLAMA_LAYOUT["metadata"],
            "tensors": [
                {**t, "group": (
                    "embedding" if t["name"].startswith("token_embd.")
                    else "output" if t["name"].startswith("output.")
                    else "attention" if ".attn_" in t["name"]
                    else "feed_forward"
                ), "block": (
                    int(t["name"].split(".")[1])
                    if t["name"].startswith("blk.") else None
                )}
                for t in LLAMA_LAYOUT["tensors"]
            ],
        }
        proposed = propose_mapping_candidates(layout, n_gpu_layers=2)
        self.assertEqual(proposed["architecture"], "llama")
        self.assertEqual(proposed["inventory_tensor_count"], 5)
        self.assertEqual(proposed["unknown_baseline_tensor_count"], 0)
        ids = {c["id"] for c in proposed["candidates"]}
        self.assertIn("single:gpu_to_host:output.weight", ids)
        self.assertIn("single:host_to_gpu:token_embd.weight", ids)
        self.assertFalse(any(c["kind"] == "tied_input_output"
                             for c in proposed["candidates"]))

    def test_arm_only_variant_is_visible_but_never_cuda_tested(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
            file.write_bytes(b"placeholder")
            models = [
                {
                    "filename": "Meta-Llama-3.1-8B-Instruct-Q4_0_4_4.gguf",
                    "quantization": "Q4_0_4_4", "quality_priority": 500,
                    "cuda_baseline_eligible": False,
                    "file_bytes": 1000,
                },
                {
                    "filename": file.name,
                    "quantization": "Q4_K_M", "quality_priority": 1000,
                    "cuda_baseline_eligible": True,
                    "file_bytes": 1000,
                },
            ]
            with (
                patch("calibration.baseline_discovery.GPUManager.calibration_status",
                      return_value=gpu_status()),
                patch("calibration.baseline_discovery.inventory",
                      return_value=models),
                patch("calibration.baseline_discovery.inspect_gguf",
                      return_value=LLAMA_LAYOUT),
                patch("calibration.baseline_discovery.probe_gpu_candidate",
                      return_value={
                          "success": True,
                          "architecture": "llama",
                          "full_gpu_native_verification": {"verified": True},
                          "measurements": {"minimal_inference_completed": True},
                      }),
                patch("calibration.baseline_discovery.remote_header") as header,
            ):
                report = choose_baseline(root=Path(tmp),
                                         model_id="llama-3.1-8b-instruct")
            self.assertTrue(report["success"])
            self.assertEqual(report["attempts"][0]["status"],
                             "excluded_publisher_arm_prepacked_format")
            self.assertEqual(report["attempts"][1]["status"],
                             "selected_full_gpu_baseline")
            header.assert_not_called()

    def test_qwen_catalog_and_cuda_baseline_inventory(self):
        names = [
            "Qwen2.5-7B-Instruct-f16.gguf",
            "Qwen2.5-7B-Instruct-Q6_K.gguf",
            "Qwen2.5-7B-Instruct-Q5_K_L.gguf",
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "Qwen2.5-7B-Instruct-Q4_0_4_4.gguf",
        ]
        with patch("huggingface_hub.HfApi") as api:
            api.return_value.model_info.return_value.siblings = [
                SimpleNamespace(rfilename=name, size=1000) for name in names
            ]
            catalog = inventory("bartowski/Qwen2.5-7B-Instruct-GGUF")
        self.assertEqual(len(catalog), len(names))
        self.assertEqual(catalog[0]["quantization"], "f16")
        self.assertEqual(catalog[1]["quantization"], "Q6_K")
        self.assertTrue(all(item["quality_priority"] is not None for item in catalog))
        self.assertFalse(next(
            item for item in catalog if item["quantization"] == "Q4_0_4_4"
        )["cuda_baseline_eligible"])

    def test_qwen_independent_output_and_chat_template(self):
        layout = {
            "metadata": {"general.architecture": "qwen2", "qwen2.block_count": 28},
            "tensors": [
                {"name": "token_embd.weight", "bytes": 10},
                {"name": "output.weight", "bytes": 10},
                {"name": "output_norm.weight", "bytes": 10},
                {"name": "blk.0.attn_q.weight", "bytes": 10},
            ],
        }
        _, expected = all_gpu_rules(layout)
        self.assertEqual(expected["expected_native_log_occurrences"], {})
        self.assertEqual(expected["output_weight_target"], "gpu")
        logs = "\n".join(
            f"tensor {item['name']} (1 MiB Q5) buffer type overridden to CUDA0"
            for item in layout["tensors"]
        )
        self.assertTrue(native_confirmation(layout, logs)["verified"])
        self.assertFalse(native_confirmation(
            layout, logs.replace("tensor output.weight (1 MiB Q5) buffer "
                                 "type overridden to CUDA0",
                                 "tensor output.weight (1 MiB Q5) buffer "
                                 "type overridden to CPU")
        )["verified"])
        prompt = minimal_chat_prompt("qwen2")
        self.assertIn("<|im_start|>user\n", prompt)
        self.assertIn("<|im_start|>assistant\n", prompt)
        self.assertIn("<|im_end|>", prompt)
        self.assertNotIn("<|start_header_id|>", prompt)

    def test_qwen_manifest_is_independent_of_other_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "qwen-2.5-7b-instruct"
            model_dir.mkdir()
            f = model_dir / "Qwen2.5-7B-Instruct-Q5_K_M.gguf"
            f.write_bytes(b"mock")
            layout = {
                "metadata": {"general.architecture": "qwen2", "qwen2.block_count": 28},
                "tensors": [
                    {"name": "token_embd.weight", "bytes": 25},
                    {"name": "output.weight", "bytes": 25},
                    {"name": "blk.0.ffn_up.weight", "bytes": 25},
                ],
                "weights_bytes": 75,
            }
            records = [{
                **catalog_entry("Q5_K_M", 700),
                "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
                "filename": f.name,
            }]
            fake = {
                "success": True, "architecture": "qwen2",
                "full_gpu_native_verification": {"verified": True},
                "measurements": {"minimal_inference_completed": True},
            }
            with (
                patch("calibration.baseline_discovery.GPUManager.calibration_status",
                      return_value=gpu_status()),
                patch("calibration.baseline_discovery.inventory", return_value=records),
                patch("calibration.baseline_discovery.inspect_gguf", return_value=layout),
                patch("calibration.baseline_discovery.probe_gpu_candidate",
                      return_value=fake),
            ):
                result = choose_baseline(
                    root=model_dir, model_id="qwen-2.5-7b-instruct",
                )
            self.assertTrue(result["success"])
            self.assertEqual(result["baseline"]["architecture"], "qwen2")
            self.assertEqual(result["baseline"]["quantization"], "Q5_K_M")
            manager = ModelManager(root=Path(tmp))
            self.assertIsNotNone(manager.selected_baseline("qwen-2.5-7b-instruct"))
            self.assertIsNone(manager.selected_baseline("gemma-2-9b-it"))
            self.assertIsNone(manager.selected_baseline("llama-3.1-8b-instruct"))

    def test_mismatched_repository_cannot_select_wrong_model(self):
        with self.assertRaises(ValueError):
            choose_baseline(model_id="llama-3.1-8b-instruct",
                            repo_id="bartowski/gemma-2-9b-it-GGUF")


if __name__ == "__main__":
    unittest.main()
