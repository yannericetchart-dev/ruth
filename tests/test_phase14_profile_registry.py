"""Local memory profile provenance, unknown-field preservation and growth alerts."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from ruth_runtime.dependencies import PROJECT_ROOT
from ruth_runtime.install_plan import WHEEL_VERSION
from ruth_runtime.model_catalog import select_model
from ruth_runtime.profile_registry import (
    record_local_profile, initialize_model_profile,
    record_minimal_context_inference, SCHEMA,
)

UUID = "GPU-5a841828-882b-85d9-c98e-3a0d29e66e00"


def hw():
    return {
        "status": "hardware_eligible_native_backend_unverified",
        "gpu_uuid": UUID, "architecture": "ada", "compute_capability": "8.9",
        "tensor_core_generation": 4, "driver_version": "595.99.02",
        "driver_cuda_max": "13.2", "backend_cuda": "13.2",
        "toolkit_cuda": "13.2",
    }


def evidence():
    choice = select_model("production")
    return {
        "action": "models", "candidate": "gemma-2-9B-it/gguf/Q4_K_L",
        "model_verified": True, "model_sha256": choice.expected_sha256,
        "source_repository": choice.repo_id, "source_revision": choice.revision,
        "native_binding_version": WHEEL_VERSION,
        "status": "gpu_load_only_failed", "mode": "weights",
        "gpu_uuid": UUID, "inference_executed": False, "generated_tokens": 0,
        "weights_loaded": False, "context_initialized": False,
        "native_cuda_buffer_mib_reported": 5700.31,
        "native_buffers": [], "cuda_free_mib_after_primary_ctx_init": 5682.81,
        "gpu_preflight": {
            "status": "gpu_ready", "gpu_uuid": UUID, "driver_version": "595.99.02",
            "processes": [], "memory_mib": {
                "total": 6141, "free": 5773, "reserved": 369, "used": 0,
            },
        },
    }


class ProfileRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "catalog").mkdir()
        shutil.copyfile(
            PROJECT_ROOT / "catalog/model_catalog.json",
            self.root / "catalog/model_catalog.json",
        )
        shutil.copyfile(
            PROJECT_ROOT / "catalog/llama_kv_backend_policy.json",
            self.root / "catalog/llama_kv_backend_policy.json",
        )
        self.evidence = self.root / "report.json"

    def store(self, report=None):
        self.evidence.write_text(json.dumps(evidence() if report is None else report))
        return record_local_profile(
            self.evidence, project_root=self.root, hardware=hw())

    def test_remote_pinned_metadata_creates_only_unknown_profile(self):
        choice = select_model("production")
        meta = {
            "status": "source_metadata_verified_license_pending",
            "repo_id": choice.repo_id, "sha256": choice.expected_sha256,
            "revision": choice.revision, "size_bytes": choice.expected_size_bytes,
        }
        result = initialize_model_profile(choice, meta, project_root=self.root)
        self.assertEqual(result["status"], "local_profile_initialized_unknown")
        obj = json.loads(Path(result["profile_file"]).read_text())
        self.assertTrue(obj["source_verified"])
        self.assertFalse(obj["model_license_accepted"])
        self.assertFalse(obj["model_locally_verified"])
        self.assertIsNone(obj["native_cuda_weights_mib"])
        self.assertEqual(obj["runtime_measurements"], {})
        second = initialize_model_profile(choice, meta, project_root=self.root)
        self.assertEqual(second["status"], "local_profile_already_exists")
        bad = dict(meta, revision="0"*40)
        with self.assertRaises(ValueError):
            initialize_model_profile(choice, bad, project_root=self.root)

    def test_first_observation_creates_profile_and_preserves_unknowns(self):
        result = self.store()
        self.assertEqual(result["status"], "local_profile_updated")
        self.assertFalse(result["git_updated"])
        path = Path(result["profile_file"])
        self.assertTrue(path.is_file())
        obj = json.loads(path.read_text())
        self.assertEqual(obj["schema"], SCHEMA)
        self.assertEqual(obj["native_cuda_weights_mib"], 5700.31)
        self.assertEqual(obj["context_profiles"], {})
        self.assertFalse(obj["production_qualified"])
        row = next(iter(obj["runtime_measurements"].values()))
        self.assertEqual(row["cuda_init_observed_nvml_delta_mib_max"], 90.19)
        self.assertEqual(row["driver_reserved_mib_max"], 369)
        self.assertIsNone(row["kv_buffer_mib"])
        self.assertIsNone(row["flash_attention"])

    def test_same_evidence_is_not_counted_twice_for_git_retry(self):
        original = self.store()
        repeated = self.store()
        self.assertEqual(repeated["status"], "local_profile_already_recorded")
        self.assertFalse(repeated["observation_duplicated"])
        obj = json.loads(Path(original["profile_file"]).read_text())
        self.assertEqual(
            next(iter(obj["runtime_measurements"].values()))["observations"], 1
        )
        self.assertEqual(len(obj["observed_evidence_sha256s"]), 1)

    def test_repeat_updates_max_and_emits_growth_warning(self):
        first = self.store()
        more = evidence()
        more["native_cuda_buffer_mib_reported"] = 5710.0
        more["gpu_preflight"]["memory_mib"]["reserved"] = 371
        second = self.store(more)
        self.assertIn("native_weights_allocation_exceeds_profiled_max",
                      second["measurement_warnings"])
        self.assertIn("driver_reservation_increased", second["measurement_warnings"])
        obj = json.loads(Path(first["profile_file"]).read_text())
        self.assertEqual(obj["native_cuda_weights_mib"], 5710.0)
        self.assertEqual(next(iter(obj["runtime_measurements"].values()))["observations"], 2)

    def test_reject_unverified_model_and_wrong_pinned_hash(self):
        broken = evidence()
        broken["model_verified"] = False
        with self.assertRaises(ValueError):
            self.store(broken)
        broken = evidence()
        broken["model_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.store(broken)

    def test_reject_concurrent_gpu(self):
        broken = evidence()
        broken["gpu_preflight"]["processes"] = [{"pid": 123, "type": "G"}]
        with self.assertRaises(ValueError):
            self.store(broken)

    def test_reject_unsupported_hardware(self):
        self.evidence.write_text(json.dumps(evidence()))
        unsupported = hw()
        unsupported["status"] = "hardware_rejected_or_unverified"
        with self.assertRaises(ValueError):
            record_local_profile(self.evidence, project_root=self.root,
                                 hardware=unsupported)

    def test_verified_minimal_inference_is_stored_without_production_claim(self):
        first = self.store()
        choice = select_model("production")
        inference = {
            "candidate": "gemma-2-9B-it/gguf/Q4_K_L",
            "gpu_inference_verified": True,
            "full_gpu_offload_verified": True,
            "gpu_process_observed": True,
            "model_verified": True, "native_library_verified": True,
            "model_loaded": True, "cpu_fallback_selected": False,
            "context_calibration_pending": True,
            "generated_tokens": 8, "n_ctx": 128,
            "n_batch": 64, "n_ubatch": 64,
            "offloaded_layers": 43, "total_layers": 43,
            "gpu_uuid": UUID,
            "gpu_preflight": {"driver_version": "595.99.02"},
        }
        status = record_minimal_context_inference(
            Path(first["profile_file"]), inference, choice=choice, hardware=hw())
        self.assertEqual(status["status"], "local_minimal_gpu_inference_verified")
        profile = json.loads(Path(first["profile_file"]).read_text())
        self.assertEqual(len(profile["minimal_context_inference"]), 1)
        self.assertFalse(profile["minimal_context_inference"][0]["production_context_qualified"])
        self.assertFalse(profile["production_qualified"])
        record_minimal_context_inference(
            Path(first["profile_file"]), inference, choice=choice, hardware=hw())
        self.assertEqual(len(json.loads(Path(first["profile_file"]).read_text())
                             ["minimal_context_inference"]), 1)
        inference["gpu_inference_verified"] = False
        with self.assertRaises(ValueError):
            record_minimal_context_inference(
                Path(first["profile_file"]), inference, choice=choice, hardware=hw())

    def test_option_specific_native_kv_profile_remains_separate_from_derived(self):
        observed = evidence()
        observed.update({
            "status": "gpu_load_only_completed",
            "mode": "context", "weights_loaded": True,
            "context_initialized": True, "n_ctx": 128,
            "n_batch": 64, "n_ubatch": 64,
            "native_buffers": [
                {"kind": "model", "device": "CUDA0", "mib": 5488.4},
                {"kind": "kv", "device": "CUDA0", "mib": 84.0},
                {"kind": "compute", "device": "CUDA0", "mib": 63.38},
            ],
            "context_options_requested": {
                "n_ctx": 128, "n_batch": 64, "n_ubatch": 64,
                "swa_full": True, "flash_attention": "off",
                "type_k": "F16", "type_v": "F16",
                "n_seq_max": 1, "kv_unified": False, "offload_kqv": True,
            },
            "native_kv_log": {
                "n_ctx_effective_native": 256,
                "n_ctx_seq_effective_native": 256,
                "n_batch_effective_native": 64,
                "n_ubatch_effective_native": 64,
                "flash_attention_effective": None,
                "global_kv": {
                    "cells": 256, "native_logical_kv": {
                        "key_logical_mib_native": 21.0,
                        "value_logical_mib_native": 21.0},
                },
                "swa_kv": {
                    "cells": 256, "native_logical_kv": {
                        "key_logical_mib_native": 21.0,
                        "value_logical_mib_native": 21.0},
                },
            },
        })
        saved = self.store(observed)
        profile = json.loads(Path(saved["profile_file"]).read_text())
        self.assertEqual(len(profile["context_profiles"]), 1)
        key, ctx = next(iter(profile["context_profiles"].items()))
        self.assertIn("/flash=off/k=F16/v=F16/swa=on/batch=64/ubatch=64", key)
        self.assertEqual(ctx["gpu_kv_buffer_mib"], 84.0)
        self.assertEqual(ctx["gpu_compute_buffer_mib"], 63.38)
        self.assertEqual(ctx["native_config"]["global_key_logical_mib_native"], 21)
        self.assertEqual(ctx["kv_source_derivation"]["key_logical_bytes_derived"],
                         42 * 1048576)
        self.assertTrue(ctx["kv_source_derivation"]["model_geometry_verification_pending"])
        observed["native_kv_log"]["global_kv"]["cells"] = 512
        with self.assertRaisesRegex(ValueError, "capacity disagrees"):
            self.store(observed)

    def test_context_remains_unknown_until_explicitly_observed(self):
        self.store()
        obj = json.loads(next(
            (self.root / ".ruth-build/profiles/models").glob("*.json")
        ).read_text())
        self.assertEqual(obj["context_profiles"], {})
        self.assertIsNone(next(iter(obj["runtime_measurements"].values()))["n_ctx"])


if __name__ == "__main__":
    unittest.main()
