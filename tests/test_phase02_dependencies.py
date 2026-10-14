"""Dependency diagnostics tests use pure-Python mocks, no real GPU needed."""
from dataclasses import replace
from importlib import metadata
import subprocess
import unittest

from ruth_runtime.dependencies import _run_probe, inspect_dependencies
from ruth_runtime.hardware import HostCPU, HostMemory, HardwareReport
from ruth_runtime.contracts import DeviceInfo


def fixture(*, supported=True, gpu=True):
    return HardwareReport(
        "Linux", "test-kernel", "x86_64", supported,
        HostCPU("x86_64", "intel", "test", 4, 2, ("avx2",)),
        HostMemory(8 * 1024**3, 5 * 1024**3, 0),
        (DeviceInfo("pci:0000:01:00.0", "gpu", "nvidia", "test", driver="580.0", uuid="GPU-test"),)
        if gpu else (),
        None, (),
    )


def installed(_):
    return "0.3.36"


def success(argv, **kwargs):
    return subprocess.CompletedProcess(argv, 0, "RUTH_OFFLOAD=yes\n", "")


def pretend_which(name):
    return "/usr/bin/" + name


class DependencyTests(unittest.TestCase):
    def test_verified_binding_is_not_inference_verification(self):
        report = inspect_dependencies(fixture(), package_version=installed,
                                      which=pretend_which, run=success,
                                      python_version=(3, 12, 4))
        self.assertTrue(report.ready_for_next_probe)
        self.assertFalse(report.to_dict()["inference_verified"])
        self.assertTrue(all(c.ok for c in report.checks if c.required_now))

    def test_missing_package_has_install_procedure_and_no_import_probe(self):
        def missing(_):
            raise metadata.PackageNotFoundError("llama-cpp-python")
        report = inspect_dependencies(fixture(), package_version=missing,
                                      which=pretend_which, run=lambda *a, **k: self.fail("must not probe"))
        self.assertFalse(report.ready_for_next_probe)
        check = next(c for c in report.checks if c.name == "llama-cpp-python")
        self.assertEqual(check.status, "missing")
        self.assertIn("venv", check.remediation)
        self.assertNotIn("llama.cpp GPU offload", [c.name for c in report.checks])

    def test_cpu_only_binding_fails_no_silent_fallback(self):
        def no_offload(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, "RUTH_OFFLOAD=no\n", "")
        report = inspect_dependencies(fixture(), package_version=installed,
                                      which=pretend_which, run=no_offload)
        self.assertEqual(next(c.status for c in report.checks if c.name == "llama.cpp GPU offload"),
                         "incompatible")
        self.assertFalse(report.ready_for_next_probe)

    def test_crash_native_probe_classified_not_success(self):
        def crash(argv, **kwargs):
            return subprocess.CompletedProcess(argv, -11, "", "segfault")
        report = inspect_dependencies(fixture(), package_version=installed,
                                      which=pretend_which, run=crash)
        self.assertFalse(report.ready_for_next_probe)
        self.assertEqual(next(c.status for c in report.checks if c.name == "llama.cpp GPU offload"),
                         "probe_failed")

    def test_no_nvidia_driver_blocks(self):
        report = inspect_dependencies(fixture(gpu=False), package_version=installed,
                                      which=pretend_which, run=success)
        self.assertFalse(report.ready_for_next_probe)
        self.assertEqual(next(c.status for c in report.checks if c.name == "GPU/driver NVIDIA"), "missing")

    def test_optional_build_tools_do_not_block_existing_wheel(self):
        def limited(name):
            return "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
        report = inspect_dependencies(fixture(), package_version=installed,
                                      which=limited, run=success)
        self.assertTrue(report.ready_for_next_probe)
        self.assertTrue(all(c.status == "optional_missing" for c in report.checks if not c.required_now))
        self.assertTrue(all(c.remediation for c in report.checks if not c.required_now))

    def test_python_or_platform_incompatible(self):
        for h, py in ((fixture(supported=False), (3, 12, 0)), (fixture(), (3, 9, 19))):
            with self.subTest(hardware=h, py=py):
                r = inspect_dependencies(h, python_version=py, package_version=installed,
                                         which=pretend_which, run=success)
                self.assertFalse(r.ready_for_next_probe)

    def test_timeout_and_unknown_output_are_not_accepted(self):
        def timeout(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 10)
        self.assertEqual(_run_probe(timeout, "python")[0], "probe_failed")
        def nonsense(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, "unexpected", "")
        self.assertEqual(_run_probe(nonsense, "python")[0], "probe_failed")

    def test_no_automatic_install_side_effect(self):
        calls = []
        def recorder(argv, **kwargs):
            calls.append(argv)
            return success(argv, **kwargs)
        inspect_dependencies(fixture(), package_version=installed,
                             which=pretend_which, run=recorder)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "-c")
        self.assertNotIn("pip", calls[0])


if __name__ == "__main__":
    unittest.main()
