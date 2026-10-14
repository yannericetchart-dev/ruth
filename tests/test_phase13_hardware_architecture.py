"""Offline Ruth architecture support and NVIDIA Toolkit/driver compatibility."""
import unittest
from types import SimpleNamespace

from ruth_runtime.hardware_eligibility import (
    classify_hardware, load_architecture_policy, inspect_host_hardware,
)

POLICY = load_architecture_policy()


def check(cc="8.9", vram=6141, driver_cuda="13.2", drv="595.99.02",
          toolkit="13.2", backend="13.2"):
    return classify_hardware(
        compute_capability=cc, total_mib=vram, driver_version=drv,
        driver_cuda_max=driver_cuda, toolkit_cuda=toolkit,
        backend_cuda=backend, policy=POLICY,
    )


class NvidiaArchitecturePolicyTests(unittest.TestCase):
    def test_ada_rtx4050_verified_and_not_latest_warning(self):
        p = check()
        self.assertEqual(p["status"], "hardware_eligible_native_backend_unverified")
        self.assertEqual(p["vram_class_gib"], 6)
        self.assertEqual(p["architecture"], "ada")
        self.assertEqual(p["tensor_core_generation"], 4)
        self.assertIn("driver_max_cuda_older_than_latest_reviewed", p["warnings"])
        self.assertFalse(p["backend_kernel_architecture_verified"])

    def test_blackwell_ws_and_dc_are_not_interchangeable(self):
        ws = check(cc="12.0", vram=96*1024, driver_cuda="13.4",
                   drv="615.10", toolkit="13.4", backend="13.4")
        self.assertEqual(ws["architecture"], "blackwell-ws")
        self.assertEqual(ws["vram_class_gib"], 96)
        self.assertEqual(ws["tensor_core_generation"], 5)
        dc = check(cc="10.0", vram=180*1024, driver_cuda="13.4",
                   drv="615.10", toolkit="13.4", backend="13.4")
        self.assertEqual(dc["architecture"], "blackwell-dc")
        self.assertEqual(dc["vram_class_gib"], 180)
        self.assertEqual(dc["min_cuda"], "12.8")

    def test_hopper_80g(self):
        p = check(cc="9.0", vram=80*1024)
        self.assertEqual(p["architecture"], "hopper")
        self.assertEqual(p["vram_class_gib"], 80)

    def test_rejects_old_gpu_and_unsupported_memory(self):
        self.assertIn("unsupported_or_unverified_compute_capability",
                      check(cc="8.6")["errors"])
        self.assertIn("vram_class_not_allowed_or_unverified",
                      check(cc="8.9", vram=4096)["errors"])
        self.assertIn("vram_class_not_allowed_or_unverified",
                      check(cc="9.0", vram=6*1024)["errors"])

    def test_blackwell_cuda_12_0_is_not_enough(self):
        p = check(cc="12.0", vram=16*1024, driver_cuda="12.0",
                  drv="525.60.13", toolkit="12.0", backend="12.0")
        self.assertIn("driver_cuda_below_architecture_minimum", p["errors"])
        self.assertIn("toolkit_cuda_below_architecture_minimum", p["errors"])
        self.assertIn("backend_cuda_below_architecture_minimum", p["errors"])

    def test_driver_must_cover_backend_and_toolkit(self):
        p = check(driver_cuda="12.8", toolkit="13.2", backend="13.2",
                  drv="570.10")
        self.assertIn("driver_cuda_below_backend_cuda", p["errors"])
        self.assertIn("driver_cuda_below_installed_toolkit", p["errors"])

    def test_newer_minor_cuda_may_have_limited_driver_compatibility(self):
        p = check(driver_cuda="13.2", toolkit="13.4", backend="13.4",
                  drv="595.99.02")
        self.assertEqual(p["status"],
                         "hardware_eligible_native_backend_unverified")
        self.assertIn("backend_newer_cuda_minor_than_driver_feature_compatibility_unverified",
                      p["warnings"])
        self.assertIn("toolkit_newer_cuda_minor_than_driver_feature_compatibility_unverified",
                      p["warnings"])
        self.assertFalse(p["backend_kernel_architecture_verified"])

    def test_old_cuda_above_arch_min_only_warns(self):
        p = check(toolkit="12.8", backend="12.8", driver_cuda="12.8",
                  drv="570.26")
        self.assertEqual(p["status"], "hardware_eligible_native_backend_unverified")
        self.assertIn("installed_toolkit_older_than_latest_reviewed", p["warnings"])

    def test_no_toolkit_is_unknown_not_fabricated(self):
        p = check(toolkit=None)
        self.assertEqual(p["status"], "hardware_eligible_native_backend_unverified")
        self.assertIsNone(p["toolkit_cuda"])
        self.assertIn("nvcc_toolkit_not_detected_backend_version_checked_separately",
                      p["warnings"])

    def test_unknown_software_versions_do_not_qualify(self):
        p = check(backend=None)
        self.assertEqual(p["status"], "hardware_rejected_or_unverified")
        self.assertIn("backend_cuda_version_unverified", p["errors"])

    def test_read_only_inspector_checks_both_driver_and_nvcc(self):
        seen = []
        def run(args, **kw):
            seen.append(args)
            if len(args) > 1 and args[0] == "nvidia-smi" and "--query-gpu=" in args[1]:
                output = ("GPU-5a841828-882b-85d9-c98e-3a0d29e66e00, "
                          "00000000:01:00.0, NVIDIA RTX 4050, 8.9, 6141, 595.99.02")
            elif args[0] == "nvidia-smi":
                output = "CUDA Version: 13.2"
            else:
                output = "Cuda compilation tools, release 13.2, V13.2.0"
            return SimpleNamespace(stdout=output, stderr="", returncode=0)
        p = inspect_host_hardware(run=run, which=lambda x: "/usr/bin/nvcc")
        self.assertEqual(p["architecture"], "ada")
        self.assertEqual(p["total_mib"], 6141)
        self.assertEqual(p["toolkit_cuda"], "13.2")
        self.assertEqual(len(seen), 3)
        self.assertTrue(p["telemetry_read_only"])


if __name__ == "__main__":
    unittest.main()
