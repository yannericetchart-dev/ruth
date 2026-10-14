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
    record_local_profile, initialize_model_profile, SCHEMA,
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

    def test_context_remains_unknown_until_explicitly_observed(self):
        self.store()
        obj = json.loads(next(
            (self.root / ".ruth-build/profiles/models").glob("*.json")
        ).read_text())
        self.assertEqual(obj["context_profiles"], {})
        self.assertIsNone(next(iter(obj["runtime_measurements"].values()))["n_ctx"])


if __name__ == "__main__":
    unittest.main()
