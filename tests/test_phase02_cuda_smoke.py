"""CUDA kernel smoke tests: entirely mocked, no NVIDIA hardware needed."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from ruth_runtime.contracts import DeviceInfo
from ruth_runtime.cuda_smoke import cuda_smoke, interpret_kernel_output
from ruth_runtime.dependencies import DependencyCheck, DependencyReport
from ruth_runtime.hardware import HardwareReport, HostCPU, HostMemory
from ruth_runtime.toolkit import ToolkitCheck


CHECKS = (
    "GNU/Linux x86_64", "Python >=3.10 (Ruth .venv)",
    "Ruth .venv", "nvidia-smi", "GPU/driver NVIDIA",
    "CUDA supportata dal driver",
)
TOKEN = "RUTH_CUDA_SMOKE_OK device=0 arch=8.9 pci=0000:01:00.0 elements=256\n"


def hardware(*, pci="0000:01:00.0", gpu_count=1):
    return HardwareReport(
        "Linux", "test-kernel", "x86_64", True,
        HostCPU("x86_64", "intel", "test", 8, 4, ("avx2",)),
        HostMemory(8 * 1024**3, 4 * 1024**3, 0),
        tuple(
            DeviceInfo("pci:" + pci, "gpu", "nvidia", "RTX 4050 Laptop",
                       pci_address=pci, driver="595.99.02",
                       uuid="GPU-simulated")
            for _ in range(gpu_count)
        ),
        None, (),
    )


def dependencies(*, failed=None):
    return DependencyReport(tuple(
        DependencyCheck(name, True, "missing" if name == failed else "ok", "simulated")
        for name in CHECKS
    ) + (DependencyCheck("llama-cpp-python", True, "missing",
                         "not yet installed"),))


def toolkit(*, present=True):
    return ToolkitCheck(
        status="ok" if present else "missing", expected_version="13.2",
        detected_version="13.2" if present else None,
        nvcc_path="/usr/local/cuda-13.2/bin/nvcc" if present else None,
        cmake_path="/usr/bin/cmake", cpp_compiler_path="/usr/bin/c++",
        cuda_architecture="89", native_build_prerequisites=present,
        detail="toolkit mock", remediation="install toolkit" if not present else None,
    )


class CudaSmokeTests(unittest.TestCase):
    def test_token_validation(self):
        self.assertTrue(interpret_kernel_output(TOKEN, "0000:01:00.0")[0])
        self.assertFalse(interpret_kernel_output("", "0000:01:00.0")[0])
        self.assertFalse(interpret_kernel_output(TOKEN * 2, "0000:01:00.0")[0])
        self.assertFalse(interpret_kernel_output(
            TOKEN.replace("arch=8.9", "arch=8.6"), "0000:01:00.0"
        )[0])
        self.assertFalse(interpret_kernel_output(TOKEN, "0000:02:00.0")[0])

    def test_readonly_default_cannot_compile(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "simple.cu"
            src.write_text("// mock source")
            report = cuda_smoke(
                source=src, hardware=hardware(),
                dependencies=dependencies(), toolkit=toolkit(),
                runner=lambda *args, **kwargs: self.fail("must not compile"),
            )
            self.assertEqual(report.status, "ready")
            self.assertTrue(report.ready)
            self.assertFalse(report.executed)
            self.assertFalse(report.gpu_verified)

    def test_missing_toolkit_blocks_and_does_not_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "test.cu"
            src.touch()
            r = cuda_smoke(
                execute=True, source=src, hardware=hardware(),
                dependencies=dependencies(), toolkit=toolkit(present=False),
                runner=lambda *args, **kwargs: self.fail("must not execute"),
            )
            self.assertEqual(r.status, "blocked")
            self.assertFalse(r.compiled)
            self.assertIn("toolkit mock", r.error)

    def test_unverified_driver_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "test.cu"
            src.touch()
            r = cuda_smoke(
                execute=True, source=src, hardware=hardware(),
                dependencies=dependencies(failed="CUDA supportata dal driver"),
                toolkit=toolkit(),
                runner=lambda *args, **kwargs: self.fail("must not execute"),
            )
            self.assertEqual(r.status, "blocked")

    def test_requires_unambiguous_gpu_identity(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "test.cu"
            src.touch()
            r = cuda_smoke(
                execute=True, source=src, hardware=hardware(gpu_count=2),
                dependencies=dependencies(), toolkit=toolkit(),
                runner=lambda *args, **kwargs: self.fail("no compile"),
            )
            self.assertEqual(r.status, "blocked")
            self.assertIn("2", r.error)

    def test_success_requires_compile_execute_and_same_pci(self):
        calls = []
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "test.cu"
            src.write_text("// mock source")
            def runner(argv, **kwargs):
                calls.append(argv)
                if "-o" in argv:
                    output = Path(argv[argv.index("-o") + 1])
                    output.write_bytes(b"fake executable")
                    return subprocess.CompletedProcess(argv, 0, "", "")
                return subprocess.CompletedProcess(argv, 0, TOKEN, "")
            r = cuda_smoke(execute=True, source=src, hardware=hardware(),
                           dependencies=dependencies(), toolkit=toolkit(),
                           runner=runner)
            self.assertEqual(r.status, "verified", r.to_dict())
            self.assertTrue(r.gpu_verified)
            self.assertTrue(r.compiled and r.executed)
            self.assertEqual(len(calls), 2)
            self.assertIn("-arch=sm_89", calls[0])
            self.assertEqual(calls[0][0], "/usr/local/cuda-13.2/bin/nvcc")
            self.assertFalse(Path(calls[0][calls[0].index("-o") + 1]).exists())

    def test_compilation_failure_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "test.cu"
            src.touch()
            r = cuda_smoke(
                execute=True, source=src, hardware=hardware(),
                dependencies=dependencies(), toolkit=toolkit(),
                runner=lambda cmd, **kw: subprocess.CompletedProcess(
                    cmd, 1, "", "nvcc mocked compilation failure"),
            )
            self.assertEqual(r.status, "compile_failed")
            self.assertFalse(r.executed)
            self.assertIn("compilation failure", r.error)

    def test_execution_failure_keeps_kernel_unverified(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "test.cu"
            src.touch()
            def run(argv, **kw):
                if "-o" in argv:
                    Path(argv[argv.index("-o") + 1]).touch()
                    return subprocess.CompletedProcess(argv, 0, "", "")
                return subprocess.CompletedProcess(argv, 17, "", "CUDA_ERROR mock")
            r = cuda_smoke(execute=True, source=src, hardware=hardware(),
                           dependencies=dependencies(), toolkit=toolkit(),
                           runner=run)
            self.assertEqual(r.status, "execution_failed")
            self.assertFalse(r.gpu_verified)
            self.assertIn("CUDA_ERROR", r.error)

    def test_pci_mismatch_rejected_despite_kernel_success(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "test.cu"
            src.touch()
            def run(argv, **kw):
                if "-o" in argv:
                    Path(argv[argv.index("-o") + 1]).touch()
                    return subprocess.CompletedProcess(argv, 0, "", "")
                return subprocess.CompletedProcess(
                    argv, 0, TOKEN.replace("01:00", "02:00"), "")
            r = cuda_smoke(execute=True, source=src, hardware=hardware(),
                           dependencies=dependencies(), toolkit=toolkit(),
                           runner=run)
            self.assertEqual(r.status, "verification_failed")
            self.assertFalse(r.gpu_verified)
            self.assertIn("PCI", r.error)

    def test_subprocess_timeout_captured(self):
        def fail(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], 120)
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "test.cu"
            src.touch()
            r = cuda_smoke(execute=True, source=src, hardware=hardware(),
                           dependencies=dependencies(), toolkit=toolkit(),
                           runner=fail)
            self.assertEqual(r.status, "probe_failed")
            self.assertFalse(r.gpu_verified)


if __name__ == "__main__":
    unittest.main()
