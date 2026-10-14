"""Auto-production GGUF selection: quality, published OOM, guarded first unknown."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from ruth_runtime.dependencies import PROJECT_ROOT
from ruth_runtime.model_catalog import select_model
from ruth_runtime.production_selection import (
    plan_production, execute_production_candidate,
)


def host():
    return {
        "status": "hardware_eligible_native_backend_unverified",
        "gpu_uuid": "GPU-5a841828-882b-85d9-c98e-3a0d29e66e00",
        "gpu_name": "NVIDIA RTX 4050 Laptop",
        "architecture": "ada", "compute_capability": "8.9",
        "driver_version": "595.99.02", "backend_cuda": "13.2",
        "vram_class_gib": 6, "total_mib": 6141,
    }


class ProductionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "catalog").mkdir()
        for name in ("model_catalog.json", "quantization_policy.json",
                     "nvidia_architecture_policy.json"):
            shutil.copyfile(PROJECT_ROOT / "catalog" / name,
                            self.root / "catalog" / name)
        pin = select_model("production")
        self.pin_l = pin
        self.pin_m = select_model("production", quantization="Q4_K_M")
        published = PROJECT_ROOT / "profiles" / "measurements" / pin.expected_sha256
        self.assertTrue(published.is_dir(), "Git measurement fixture must exist")
        target = self.root / "profiles" / "measurements" / pin.expected_sha256
        shutil.copytree(published, target)

    def inventory(self, rows=None):
        l = self.pin_l
        m = self.pin_m
        defaults = [
            ("Q8_0", 200, 9830000000, "1"*64, False),
            ("Q5_K_S", 150, 6480000000, "2"*64, False),
            ("Q4_K_L", 140, l.expected_size_bytes, l.expected_sha256, True),
            ("Q4_K_M", 130, m.expected_size_bytes, m.expected_sha256, True),
            ("Q4_K_S", 120, 5480000000, "3"*64, False),
            ("IQ4_XS", 110, 5180000000, "4"*64, False),
        ]
        rows = defaults if rows is None else rows
        return {
            "status": "quantizations_discovered",
            "repo_id": l.repo_id, "revision": l.revision,
            "quantizations": [{
                "quantization": q, "size_bytes": size, "sha256": sha,
                "quality_priority_heuristic": priority,
                "quality_tier_heuristic": "balanced",
                "exclusion_reason": None,
                "catalog_pin_exists": pinned, "download_pin_ready": pinned,
            } for q, priority, size, sha, pinned in rows],
        }

    def plan(self, **kwargs):
        inventory = self.inventory()
        return plan_production(project_root=self.root, hardware=host(),
                               authorize_metadata_network=True,
                               discover=lambda **kw: inventory, **kwargs)

    def test_q5_skipped_l_known_bad_q4m_first_without_native_load(self):
        with patch("subprocess.run", side_effect=AssertionError("No GPU processes")):
            result = self.plan()
        self.assertEqual(result["status"], "production_selection_ready")
        self.assertEqual(result["first_candidate_to_test"], "Q4_K_M")
        ranks = [r["quantization"] for r in result["models"]]
        self.assertEqual(ranks, ["Q8_0", "Q5_K_S", "Q4_K_L",
                                 "Q4_K_M", "Q4_K_S", "IQ4_XS"])
        states = {r["quantization"]:r["status"] for r in result["models"]}
        self.assertEqual(states["Q5_K_S"],
                         "policy_excluded_gguf_file_larger_than_host_vram")
        self.assertEqual(states["Q4_K_L"], "measured_full_gpu_weights_incompatible")
        self.assertEqual(states["Q4_K_M"],
                         "candidate_requires_download_and_gpu_tests")
        self.assertEqual(states["Q4_K_S"], "requires_catalog_pin_before_download")
        self.assertFalse(result["gpu_allocations_executed"])
        self.assertFalse(result["download_started"])

    def test_no_authorization_no_downloading_or_gpu_allocations(self):
        outcome = execute_production_candidate(plan=self.plan())
        self.assertEqual(outcome["status"], "production_explicit_authorization_required")
        self.assertFalse(outcome["gpu_allocations_executed"])
        self.assertFalse(outcome["download_started"])

    def test_verified_q4m_does_not_infer_smaller_are_runtime_verified(self):
        plan = self.plan()
        last = self.pin_m
        reports = []
        def fake_load(**kw):
            reports.append(kw["mode"])
            return {
                "action": "models", "status": "gpu_load_only_completed",
                "candidate": "gemma-2-9B-it/gguf/Q4_K_M",
                "model_sha256": last.expected_sha256,
                "model_verified": True, "weights_loaded": True,
                "context_initialized": kw["mode"] == "context",
            }
        dr = SimpleNamespace(model_verified=True, status="model_already_verified",
                             download_started=False)
        with patch("ruth_runtime.model_download.download_pinned_model",
                   side_effect=AssertionError("Use injected download")), \
             patch("ruth_runtime.profile_registry.record_local_profile",
                   side_effect=AssertionError("Use injected recorder")):
            result = execute_production_candidate(
                plan=plan, project_root=self.root,
                allow_model_download=True, acknowledge_model_license=True,
                allow_gpu_load=True, allow_gpu_inference=True,
                download=lambda *a, **kw: dr, load=fake_load,
                infer=lambda **kw: SimpleNamespace(
                    gpu_inference_verified=True, generated_tokens=8,
                    to_dict=lambda: {"gpu_inference_verified": True}),
                record=lambda evidence, **kw: {
                    "profile_file": str(evidence), "status": "local_profile_updated",
                },
                build=lambda path: {"proposal": {"verified": False}},
                publish_owner=lambda proposal: {
                    "status": "git_profile_updated_model_profiling",
                    "git_updated": True},
            )
        self.assertEqual(reports, ["weights", "context"])
        self.assertEqual(result["status"],
                         "production_candidate_min_context_gpu_verified")
        self.assertEqual(result["candidate"], "Q4_K_M")
        self.assertIn("Q4_K_S", result["smaller_formats_preliminary_size_candidates"])
        self.assertEqual(result["smaller_formats_full_gpu_verified"], [])
        self.assertFalse(result["production_qualified"])

    def test_larger_gguf_prefilter_is_not_tensor_allocation_proof(self):
        result = self.plan()
        self.assertTrue(result["gguf_size_policy_is_not_native_cuda_allocation_proof"])
        self.assertTrue(all(not r["full_gpu_inference_verified"]
                            for r in result["models"]))

    def test_missing_exact_profile_q4l_is_not_skipped(self):
        shutil.rmtree(self.root / "profiles" / "measurements" / self.pin_l.expected_sha256)
        result = self.plan()
        states = {r["quantization"]: r["status"] for r in result["models"]}
        self.assertEqual(states["Q4_K_L"],
                         "candidate_requires_download_and_gpu_tests")
        self.assertEqual(result["first_candidate_to_test"], "Q4_K_L")

    def test_incompatible_host_fails_without_metadata_lookup(self):
        result = plan_production(
            project_root=self.root,
            hardware={"status": "hardware_rejected_or_unverified"},
            authorize_metadata_network=True,
            discover=lambda **kw: self.fail("No source lookup"),
        )
        self.assertEqual(result["status"], "production_host_ineligible")

    def test_source_changed_is_blocked(self):
        inv = self.inventory()
        inv["revision"] = "a" * 40
        outcome = plan_production(
            project_root=self.root, hardware=host(),
            authorize_metadata_network=True,
            discover=lambda **kw: inv,
        )
        self.assertEqual(outcome["status"], "production_source_mismatch")


if __name__ == "__main__":
    unittest.main()
