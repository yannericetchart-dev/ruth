"""CUDA Toolkit-first preflight tests; no GPU, network or installation needed."""
from pathlib import Path
import subprocess
import unittest

from ruth_runtime.toolkit import (
    CUDA_TOOLKIT_VERSION, CUDA_ARCHITECTURE, TOOLKIT_PACKAGE,
    inspect_cuda_toolkit, parse_nvcc_version,
)


def probe(argv, **kwargs):
    return subprocess.CompletedProcess(
        argv, 0, "nvcc: NVIDIA (R) Cuda compiler driver\n"
        "Cuda compilation tools, release 13.2, V13.2.51", ""
    )


def paths(name):
    return "/usr/bin/" + name


class CudaToolkitTests(unittest.TestCase):
    def test_expected_version_and_device_architecture(self):
        self.assertEqual(CUDA_TOOLKIT_VERSION, (13, 2))
        self.assertEqual(CUDA_ARCHITECTURE, "89")
        self.assertEqual(TOOLKIT_PACKAGE, "cuda-toolkit-13-2")

    def test_parser_distinguishes_toolkit_from_smi(self):
        self.assertEqual(parse_nvcc_version("Cuda compilation tools, release 13.2, V13.2.51"),
                         (13, 2))
        self.assertIsNone(parse_nvcc_version("NVIDIA-SMI CUDA Version: 13.2"))
        self.assertIsNone(parse_nvcc_version("corrupted output"))

    def test_missing_toolkit_gives_remediation_without_probe(self):
        report = inspect_cuda_toolkit(
            which=lambda n: "/usr/bin/" + n if n != "nvcc" else None,
            exists=lambda p: False, environ={},
            run=lambda *a, **k: self.fail("Do not run when nvcc missing"),
        )
        self.assertEqual(report.status, "missing")
        self.assertFalse(report.native_build_prerequisites)
        self.assertIn("cuda-toolkit-13-2", report.remediation)
        self.assertIn("apt-get -s install", report.remediation)
        self.assertNotIn("sudo apt install cuda ", report.remediation)

    def test_native_prerequisites_detected_and_no_compile_executed(self):
        called = []
        def runner(argv, **kwargs):
            called.append(argv)
            return probe(argv, **kwargs)
        report = inspect_cuda_toolkit(
            which=paths, exists=lambda p: False,
            environ={}, run=runner,
        )
        self.assertEqual(report.status, "ok")
        self.assertTrue(report.native_build_prerequisites)
        self.assertEqual(report.detected_version, "13.2")
        self.assertEqual(called, [["/usr/bin/nvcc", "--version"]])
        self.assertIsNone(report.remediation)

    def test_explicit_132_directory_wins_over_wrong_path(self):
        reported = []
        def runner(argv, **kwargs):
            reported.append(argv)
            return probe(argv, **kwargs)
        report = inspect_cuda_toolkit(
            which=paths,
            exists=lambda p: p == Path("/usr/local/cuda-13.2/bin/nvcc"),
            environ={}, run=runner,
        )
        self.assertEqual(report.status, "ok")
        self.assertEqual(reported[0][0], "/usr/local/cuda-13.2/bin/nvcc")

    def test_wrong_nvcc_version_does_not_use_wheel(self):
        def wrong(argv, **kwargs):
            return subprocess.CompletedProcess(
                argv, 0, "Cuda compilation tools, release 13.1, V13.1.0", ""
            )
        r = inspect_cuda_toolkit(which=paths, exists=lambda p: False,
                                 environ={}, run=wrong)
        self.assertEqual(r.status, "incompatible")
        self.assertFalse(r.native_build_prerequisites)
        self.assertIn("13.1", r.detail)

    def test_missing_compiler_or_cmake_reported_as_build_error(self):
        r = inspect_cuda_toolkit(
            which=lambda n: None if n == "c++" else paths(n),
            exists=lambda p: False, environ={}, run=probe,
        )
        self.assertEqual(r.status, "missing_build_tools")
        self.assertFalse(r.native_build_prerequisites)
        self.assertIn("build-essential", r.remediation)

    def test_crash_unknown_output_and_exception_are_failures(self):
        for result in (
            subprocess.CompletedProcess(["nvcc"], 7, "", "driver error"),
            subprocess.CompletedProcess(["nvcc"], 0, "unknown", ""),
        ):
            with self.subTest(result=result):
                r = inspect_cuda_toolkit(
                    which=paths, exists=lambda p: False, environ={},
                    run=lambda *args, **kwargs: result,
                )
                self.assertEqual(r.status, "probe_failed")
        def crashed(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], 8)
        r = inspect_cuda_toolkit(
            which=paths, exists=lambda p: False, environ={}, run=crashed,
        )
        self.assertEqual(r.status, "probe_failed")

    def test_cuda_home_without_path_is_supported(self):
        r = inspect_cuda_toolkit(
            which=lambda n: paths(n) if n != "nvcc" else None,
            exists=lambda p: p == Path("/opt/cuda-13.2/bin/nvcc"),
            environ={"CUDA_HOME": "/opt/cuda-13.2"}, run=probe,
        )
        self.assertEqual(r.nvcc_path, "/opt/cuda-13.2/bin/nvcc")
        self.assertTrue(r.native_build_prerequisites)


if __name__ == "__main__":
    unittest.main()
