"""Offline profile matrices, source identity, host DRM and non-executing force."""
import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruth_runtime.dependencies import PROJECT_ROOT
from ruth_runtime.install_plan import WHEEL_VERSION
from ruth_runtime.launcher import main
from ruth_runtime.model_catalog import select_model
from ruth_runtime.model_memory_profiles import (
    build_profile_matrix, inspect_gpu_drm,
)

UUID = "GPU-5a841828-882b-85d9-c98e-3a0d29e66e00"
PCI = "00000000:01:00.0"


def fake_gpu(**kw):
    return SimpleNamespace(uuid=UUID, pci_bus=PCI, total_mib=6141, free_mib=5773)


def fake_preflight(uuid):
    assert uuid == UUID
    return SimpleNamespace(
        status="gpu_ready",
        to_dict=lambda: {
            "status": "gpu_ready", "driver_version": "595.99.02",
            "gpu_uuid": UUID, "processes": [], "memory_mib": {
                "total": 6141, "free": 5773, "reserved": 369, "used": 0,
            },
        },
    )


def fake_drm(bus):
    assert bus == PCI
    return {"status": "drm_not_loaded", "module_loaded": False,
            "associated_cards": []}


def reference_evidence():
    choice = select_model("production")
    return {
        "action": "models", "candidate": "gemma-2-9B-it/gguf/Q4_K_L",
        "model_sha256": choice.expected_sha256,
        "native_binding_version": WHEEL_VERSION,
        "status": "gpu_load_only_failed",
        "mode": "weights", "inference_executed": False,
        "generated_tokens": 0, "weights_loaded": False,
        "context_initialized": False,
        "native_cuda_buffer_mib_reported": 5700.31,
        "cuda_preinit_requested": True,
        "cuda_free_mib_after_primary_ctx_init": 5682.81,
        "gpu_preflight": {
            "gpu_uuid": UUID, "driver_version": "595.99.02",
            "memory_mib": {"total": 6141, "free": 5773,
                           "reserved": 369, "used": 0},
        },
    }


class MemoryProfileMatrixTests(unittest.TestCase):
    def create_evidence(self, temp, obj=None):
        path = Path(temp) / "report.json"
        path.write_text(json.dumps(obj if obj is not None else reference_evidence()))
        return path

    def matrix(self, **kw):
        return build_profile_matrix(project_root=PROJECT_ROOT,
                                    gpu_inspector=fake_gpu,
                                    preflight=fake_preflight,
                                    drm_inspector=fake_drm, **kw)

    def test_class_listing_no_hardware_calls_or_allocations(self):
        matrix = self.matrix()
        self.assertEqual(matrix["status"], "profile_catalog_ready")
        self.assertEqual(matrix["capacity_classes_gib"][:3], [4, 6, 8])
        self.assertIsNone(matrix["host"])
        self.assertFalse(matrix["gpu_allocations_executed"])
        self.assertFalse(matrix["gpu_inference_executed"])
        self.assertFalse(matrix["download_started"])
        self.assertTrue(matrix["profiles"])
        self.assertTrue(all(not p["production_qualified"]
                            for p in matrix["profiles"]))

    def test_observed_weights_exclude_four_but_do_not_certify_eight(self):
        with TemporaryDirectory() as temp:
            m = self.matrix(evidence_file=self.create_evidence(temp))
        row = next(p for p in m["profiles"] if p["id"].endswith("/Q4_K_L"))
        classes = {p["nominal_gib"]: p for p in row["capacity_classes"]}
        self.assertEqual(row["measured_weights_buffer_mib"], 5700.31)
        self.assertEqual(row["source_evidence"]["driver_reserved_mib"], 369)
        self.assertEqual(row["source_evidence"]["observed_nvml_to_cuda_delta_mib"], 90.19)
        self.assertIsNone(row["source_evidence"]["native_kv_gpu_buffer_mib"])
        self.assertIsNone(row["source_evidence"]["native_compute_gpu_buffer_mib"])
        self.assertEqual(classes[4]["status"],
                         "insufficient_nominal_vram_for_measured_weights")
        self.assertEqual(classes[6]["status"],
                         "weights_physically_possible_unverified")
        self.assertEqual(classes[8]["status"],
                         "weights_physically_possible_unverified")
        self.assertFalse(classes[8]["full_runtime_compatible"])

    def test_exact_host_measurement_marks_insufficient(self):
        with TemporaryDirectory() as temp:
            m = self.matrix(check_host=True, evidence_file=self.create_evidence(temp))
        row = next(p for p in m["profiles"] if p["id"].endswith("/Q4_K_L"))
        self.assertEqual(row["host_status"],
                         "host_measured_weights_cannot_fit")
        self.assertTrue(m["host"]["compute_only_verified"])
        self.assertEqual(m["host"]["gpu_preflight"]["memory_mib"]["reserved"], 369)

    def test_mismatch_driver_does_not_reuse_other_machine_result(self):
        with TemporaryDirectory() as temp:
            report = reference_evidence()
            report["gpu_preflight"]["driver_version"] = "595.99.01"
            m = self.matrix(check_host=True, evidence_file=self.create_evidence(temp, report))
        row = next(p for p in m["profiles"] if p["id"].endswith("/Q4_K_L"))
        self.assertEqual(row["host_status"], "host_evidence_identity_mismatch")

    def test_missing_sha_never_creates_measured_weight_profile(self):
        with TemporaryDirectory() as temp:
            report = reference_evidence()
            report.pop("model_sha256")
            m = self.matrix(evidence_file=self.create_evidence(temp, report))
        row = next(p for p in m["profiles"] if p["id"].endswith("/Q4_K_L"))
        self.assertEqual(row["native_evidence_status"],
                         "missing_or_mismatched_gguf_sha256")
        self.assertIsNone(row["measured_weights_buffer_mib"])

    def test_force_only_marks_separate_experiment(self):
        with TemporaryDirectory() as temp:
            m = self.matrix(check_host=True, force=True,
                            evidence_file=self.create_evidence(temp))
        row = next(p for p in m["profiles"] if p["id"].endswith("/Q4_K_L"))
        self.assertTrue(row["experiment_allowed_in_separate_probe"])
        self.assertFalse(row["production_qualified"])
        self.assertFalse(m["gpu_allocations_executed"])

    def test_busy_host_cannot_be_forced(self):
        def busy(uuid):
            return SimpleNamespace(status="gpu_busy", to_dict=lambda: {
                "status": "gpu_busy", "driver_version": "595.99.02",
                "processes": [{"pid": 3032, "name": "Xorg"}],
            })
        m = build_profile_matrix(project_root=PROJECT_ROOT, check_host=True,
                                 force=True, gpu_inspector=fake_gpu,
                                 preflight=busy, drm_inspector=fake_drm)
        self.assertFalse(m["host"]["exclusivity_verified"])
        self.assertFalse(any(x["experiment_allowed_in_separate_probe"]
                             for x in m["profiles"]))

    def test_active_drm_blocks_compute_only_force_advice(self):
        def active(bus):
            return {"status": "drm_active_on_selected_gpu", "module_loaded": True,
                    "associated_cards": ["card1"]}
        m = build_profile_matrix(project_root=PROJECT_ROOT, check_host=True,
                                 force=True, gpu_inspector=fake_gpu,
                                 preflight=fake_preflight, drm_inspector=active)
        self.assertFalse(m["host"]["compute_only_verified"])
        self.assertFalse(any(p["experiment_allowed_in_separate_probe"]
                             for p in m["profiles"]))

    def test_drm_pci_bus_association(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "class/drm/card0").mkdir(parents=True)
            (root / "module").mkdir()
            hardware = root / "devices/pci0000:00/0000:01:00.0"
            hardware.mkdir(parents=True)
            (root / "class/drm/card0/device").symlink_to(hardware)
            self.assertEqual(inspect_gpu_drm(PCI, sysroot=root)["status"],
                             "drm_not_loaded")
            (root / "module/nvidia_drm").mkdir()
            d = inspect_gpu_drm(PCI, sysroot=root)
            self.assertEqual(d["status"], "drm_active_on_selected_gpu")
            self.assertEqual(d["associated_cards"], ["card0"])
            d = inspect_gpu_drm("00000000:02:00.0", sysroot=root)
            self.assertEqual(d["status"], "drm_module_loaded_no_selected_gpu_card")

    def test_cli_listing_is_offline(self):
        with patch("sys.stdout", new_callable=StringIO) as output, \
             patch("ruth_runtime.model_memory_profiles.inspect_single_nvidia_gpu",
                   side_effect=AssertionError("No hardware reads without --host")):
            self.assertEqual(main(["models", "profiles"]), 0)
        body = json.loads(output.getvalue())
        self.assertEqual(body["schema"], "ruth-model-memory-profiles-v1")
        self.assertFalse(body["gpu_allocations_executed"])


if __name__ == "__main__":
    unittest.main()
