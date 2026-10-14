"""Simulate a fresh machine: never trust global site-packages."""
from importlib import metadata
from dataclasses import replace
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest

from ruth_runtime.dependencies import _run_probe, inspect_dependencies
from ruth_runtime.requirements import (
    MIN_NVIDIA_DRIVER, MIN_CUDA_VERSION, driver_meets_policy,
    parse_driver_version, parse_reported_cuda_version, cuda_meets_policy,
)
from ruth_runtime.hardware import HostCPU, HostMemory, HardwareReport
from ruth_runtime.contracts import DeviceInfo

FAKE_ROOT = Path("/nonexistent/ruth-project")
FAKE_VENV = FAKE_ROOT / ".venv"
BOOTSTRAP = Path(__file__).resolve().parents[1] / "bootstrap.sh"


def fixture(*, supported=True, gpu=True):
    return HardwareReport(
        "Linux", "test-kernel", "x86_64", supported,
        HostCPU("x86_64", "intel", "test", 4, 2, ("avx2",)),
        HostMemory(8 * 1024**3, 5 * 1024**3, 0),
        (DeviceInfo("pci:0000:01:00.0", "gpu", "nvidia", "test",
                    driver="595.58.03", uuid="GPU-test"),) if gpu else (),
        None, (),
    )


def installed(_):
    return "0.3.36"


def success(argv, **kwargs):
    return subprocess.CompletedProcess(argv, 0, "RUTH_OFFLOAD=yes\n", "")


def fake_nvidia_smi(argv, **kwargs):
    return subprocess.CompletedProcess(argv, 0,
        "NVIDIA-SMI 595.58.03  Driver Version: 595.58.03  CUDA Version: 13.2\n", "")


def pretend_which(name):
    return "/usr/bin/" + name


def inspect(hardware=None, *, project_root=FAKE_ROOT, venv_prefix=FAKE_VENV, **kwargs):
    return inspect_dependencies(
        fixture() if hardware is None else hardware,
        project_root=project_root, venv_prefix=venv_prefix,
        python_version=kwargs.pop("python_version", (3, 12, 4)),
        package_version=kwargs.pop("package_version", installed),
        which=kwargs.pop("which", pretend_which),
        run=kwargs.pop("run", success),
        smi_run=kwargs.pop("smi_run", fake_nvidia_smi), **kwargs)


class DependencyTests(unittest.TestCase):
    def test_valid_venv_and_binding_not_inference_verification(self):
        report = inspect()
        self.assertTrue(report.ready_for_next_probe)
        self.assertFalse(report.to_dict()["inference_verified"])

    def test_global_python_cannot_satisfy_project_venv(self):
        calls = []
        def forbidden_package(_):
            calls.append("package_version")
            raise AssertionError("Host packages must never be inspected")
        def forbidden_probe(*args, **kwargs):
            calls.append("probe")
            raise AssertionError("No binding probing outside Ruth .venv")
        report = inspect(venv_prefix=Path("/some/other/env"),
                         package_version=forbidden_package, run=forbidden_probe)
        self.assertFalse(report.ready_for_next_probe)
        self.assertEqual(calls, [])
        self.assertEqual(next(c.status for c in report.checks if c.name == "Ruth .venv"), "missing")
        self.assertEqual(next(c.status for c in report.checks if c.name == "llama-cpp-python"), "blocked")

    def test_global_site_packages_venv_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / ".venv" / "pyvenv.cfg"
            cfg.parent.mkdir()
            cfg.write_text("home = /usr/bin\ninclude-system-site-packages = true\n")
            result = inspect(project_root=root, venv_prefix=root / ".venv",
                             package_version=lambda _: self.fail("global package leak"))
            self.assertFalse(result.ready_for_next_probe)
            self.assertEqual(next(c.status for c in result.checks
                                  if c.name == "Ruth .venv"), "missing")

    def test_project_venv_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "external"
            target.mkdir()
            (root / ".venv").symlink_to(target, target_is_directory=True)
            report = inspect(project_root=root, venv_prefix=root / ".venv",
                             package_version=lambda _: self.fail("unexpected package read"))
            self.assertFalse(report.ready_for_next_probe)
            self.assertEqual(next(c.status for c in report.checks
                                  if c.name == "Ruth .venv"), "missing")

    def test_missing_python_venv_has_usable_setup_instructions(self):
        report = inspect(venv_prefix=Path("/system"))
        for name in ("Ruth .venv", "Python >=3.10 (Ruth .venv)"):
            entry = next(c for c in report.checks if c.name == name)
            self.assertIn("bootstrap.sh", entry.remediation)

    def test_missing_binding_remediation_and_no_probe(self):
        def missing(_):
            raise metadata.PackageNotFoundError("llama-cpp-python")
        report = inspect(package_version=missing,
                         run=lambda *a, **k: self.fail("Do not load absent binding"))
        self.assertFalse(report.ready_for_next_probe)
        check = next(c for c in report.checks if c.name == "llama-cpp-python")
        self.assertEqual(check.status, "missing")
        self.assertIn(".venv/bin/python", check.remediation)
        self.assertNotIn("llama.cpp GPU offload", [c.name for c in report.checks])

    def test_cpu_only_binding_fails_without_fallback(self):
        def no_offload(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, "RUTH_OFFLOAD=no\n", "")
        report = inspect(run=no_offload)
        self.assertFalse(report.ready_for_next_probe)
        self.assertEqual(next(c.status for c in report.checks
                              if c.name == "llama.cpp GPU offload"), "incompatible")

    def test_crash_or_timeout_classified_as_failure(self):
        def crash(argv, **kwargs):
            return subprocess.CompletedProcess(argv, -11, "", "segfault")
        self.assertEqual(next(c.status for c in inspect(run=crash).checks
                              if c.name == "llama.cpp GPU offload"), "probe_failed")
        def timeout(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 15)
        self.assertEqual(_run_probe(timeout, "python")[0], "probe_failed")

    def test_driver_minimum_595_cuda_132_enforced(self):
        self.assertEqual(MIN_NVIDIA_DRIVER, (595, 45, 4))
        self.assertEqual(MIN_CUDA_VERSION, (13, 2))
        self.assertEqual(parse_driver_version("595.45.04"), (595, 45, 4))
        self.assertFalse(driver_meets_policy("595.0"))
        self.assertFalse(driver_meets_policy("590.99.999"))
        self.assertTrue(driver_meets_policy("595.45.04"))
        self.assertTrue(driver_meets_policy("610.0.0"))
        self.assertFalse(driver_meets_policy("garbage"))
        self.assertEqual(parse_reported_cuda_version(
            "NVIDIA-SMI ... CUDA Version: 13.2"), (13, 2))
        self.assertTrue(cuda_meets_policy((13, 2)))
        self.assertFalse(cuda_meets_policy((13, 1)))

    def test_outdated_gpu_driver_blocks_even_with_valid_binding(self):
        report = inspect(hardware=fixture())
        self.assertTrue(report.ready_for_next_probe)
        older = fixture()
        older = replace(
            older, gpus=(__import__("dataclasses").replace(
                older.gpus[0], driver="580.178.04"),))
        result = inspect(hardware=older)
        self.assertFalse(result.ready_for_next_probe)
        d = next(c for c in result.checks if c.name == "GPU/driver NVIDIA")
        self.assertEqual(d.status, "incompatible")
        self.assertIn("595.45.04", d.remediation or "")
        self.assertEqual(next(c.status for c in result.checks
                              if c.name == "CUDA supportata dal driver"), "blocked")

    def test_cuda_version_below_target_and_unreadable_fail(self):
        def old_cuda(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0,
                "NVIDIA-SMI ... CUDA Version: 13.1", "")
        def invalid_cuda(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, "unexpected", "")
        self.assertEqual(next(c.status for c in inspect(smi_run=old_cuda).checks
                              if c.name == "CUDA supportata dal driver"),
                         "incompatible")
        self.assertEqual(next(c.status for c in inspect(smi_run=invalid_cuda).checks
                              if c.name == "CUDA supportata dal driver"),
                         "probe_failed")
        self.assertFalse(inspect(smi_run=old_cuda).ready_for_next_probe)

    def test_nvidia_smi_failure_is_not_missing_driver(self):
        def reboot_pending(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 9, "", "Driver/library mismatch")
        result = inspect(smi_run=reboot_pending)
        self.assertEqual(next(c.status for c in result.checks
                              if c.name == "CUDA supportata dal driver"),
                         "probe_failed")
        self.assertFalse(result.ready_for_next_probe)

    def test_nvidia_driver_absent_blocks_even_if_binding_present(self):
        report = inspect(hardware=fixture(gpu=False))
        self.assertFalse(report.ready_for_next_probe)
        self.assertEqual(next(c.status for c in report.checks
                              if c.name == "GPU/driver NVIDIA"), "missing")

    def test_missing_optional_build_tools_do_not_block_existing_cuda_wheel(self):
        report = inspect(which=lambda name: "/usr/bin/nvidia-smi"
                         if name == "nvidia-smi" else None)
        self.assertTrue(report.ready_for_next_probe)
        self.assertTrue(all(c.status == "optional_missing"
                            for c in report.checks if not c.required_now))
        self.assertTrue(all(c.remediation for c in report.checks if not c.required_now))

    def test_python_or_platform_incompatible(self):
        for h, py in ((fixture(supported=False), (3, 12, 0)),
                      (fixture(), (3, 9, 19))):
            with self.subTest(hw=h, py=py):
                self.assertFalse(inspect(hardware=h, python_version=py).ready_for_next_probe)

    def test_no_pip_check_or_package_installations(self):
        calls = []
        def recorder(argv, **kwargs):
            calls.append(argv)
            return success(argv, **kwargs)
        report = inspect(run=recorder)
        self.assertFalse(any(c.name.lower() == "pip" for c in report.checks))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "-c")
        self.assertNotIn("install", calls[0])


@unittest.skipUnless(platform.system() == "Linux"
                     and platform.machine() == "x86_64",
                     "Bootstrap is GNU/Linux x86_64 only")
class BootstrapTests(unittest.TestCase):
    def test_read_only_plan_and_missing_venv(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "bootstrap.sh"
            shutil.copyfile(BOOTSTRAP, dest)
            plan = subprocess.run(["sh", str(dest), "--plan"],
                                  capture_output=True, text=True)
            self.assertEqual(plan.returncode, 0, plan.stderr)
            self.assertFalse((Path(td) / ".venv").exists())
            check = subprocess.run(["sh", str(dest), "--check"],
                                   capture_output=True, text=True)
            self.assertNotEqual(check.returncode, 0)
            self.assertIn("--create-venv", check.stderr)

    def test_missing_host_python_bootstrapping_scenario(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "bootstrap.sh"
            shutil.copyfile(BOOTSTRAP, dest)
            bin_dir = Path(td) / "bin"
            bin_dir.mkdir()
            for command in ("uname", "sed", "head", "tr"):
                binary = shutil.which(command)
                if binary:
                    (bin_dir / command).symlink_to(binary)
            import os
            env = dict(os.environ, PATH=str(bin_dir))
            out = subprocess.run(["/bin/sh", str(dest), "--check"],
                                 capture_output=True, text=True, env=env)
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("python3 is missing", out.stderr)
            self.assertIn("Python 3.10+", out.stdout)
            self.assertFalse((Path(td) / ".venv").exists())

    def test_create_venv_only_when_explicit(self):
        if sys.version_info < (3, 10):
            self.skipTest("Python bootstrap requires >=3.10")
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "bootstrap.sh"
            shutil.copyfile(BOOTSTRAP, dest)
            created = subprocess.run(["sh", str(dest), "--create-venv"],
                                     capture_output=True, text=True, timeout=40)
            if created.returncode != 0 and "venv/ensurepip" in created.stderr:
                self.skipTest("This test host lacks the optional venv package")
            self.assertEqual(created.returncode, 0, created.stderr)
            self.assertTrue((Path(td) / ".venv" / "pyvenv.cfg").is_file())
            verified = subprocess.run(["sh", str(dest), "--check"],
                                      capture_output=True, text=True, timeout=20)
            self.assertEqual(verified.returncode, 0, verified.stderr)


if __name__ == "__main__":
    unittest.main()
