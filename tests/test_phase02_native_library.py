"""Offline regression tests for installed CUDA C ABI provenance and sharing."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

from ruth_runtime.install_plan import SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT
from ruth_runtime.native_builder import BUILD_ROOT, BUILD_OUTPUT
from ruth_runtime.native_installer import NATIVE_LOCK, RUNTIME_FOLDER, PTH_FILE
from ruth_runtime.native_library import verify_native_library
from ruth_runtime.direct_llama import (
    DirectLlamaLibraryAdapter, DirectNativeUnavailable,
)

LIBS = {
    "libllama.so": b"mock llama dynamic elf bytes",
    "libggml-cuda.so": b"mock ggml cuda dynamic elf bytes",
    "libggml.so": b"mock ggml dynamic elf bytes",
}


def fixture(root):
    installed = root / ".venv" / RUNTIME_FOLDER
    lib = installed / "llama_cpp/lib"
    lib.mkdir(parents=True)
    for name, data in LIBS.items():
        (lib / name).write_bytes(data)
    site = root / ".venv/lib/python3.12/site-packages"
    site.mkdir(parents=True)
    (site / PTH_FILE).write_text(str(installed.resolve()) + "\n")
    build = root / BUILD_ROOT / BUILD_OUTPUT
    wh = build / "wheels"
    wh.mkdir(parents=True)
    filename = "llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl"
    wheel = wh / filename
    with ZipFile(wheel, "w") as z:
        z.writestr(
            "llama_cpp_python-0.3.36.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: llama-cpp-python\nVersion: 0.3.36\n"
        )
        for name, data in LIBS.items():
            z.writestr("llama_cpp/lib/" + name, data)
    checksum = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (root / "locks").mkdir()
    (root / NATIVE_LOCK).write_text(json.dumps({
        "schema": "ruth-native-wheel-lock-v1",
        "source_commit": SOURCE_COMMIT,
        "llama_cpp_commit": LLAMA_CPP_SUBMODULE_COMMIT,
        "python": "3.12", "cuda_architecture": "89",
        "wheel_filename": filename, "wheel_sha256": checksum,
    }))
    (build / "build-result.json").write_text(json.dumps({
        "source_commit": SOURCE_COMMIT,
        "llama_cpp_commit": LLAMA_CPP_SUBMODULE_COMMIT,
        "wheel_file": filename, "wheel_sha256": checksum,
        "ggml_cuda_library_packaged": True,
    }))
    env = SimpleNamespace(
        interpreter_in_project_venv=True, python_version="3.12.3",
        project_root=str(root), interpreter=str(root / ".venv/bin/python"),
    )
    return env, lib


class SubprocessProbe:
    def __init__(self, *, direct_gpu=True, same=True, python_gpu=True, fail=None):
        self.calls = []
        self.direct_gpu = direct_gpu
        self.same = same
        self.python_gpu = python_gpu
        self.fail = fail

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        assert argv[:2] == [str(Path(kwargs["cwd"]) / ".venv/bin/python"), "-I"]
        assert "pip" not in argv and "--no-index" not in argv
        assert "LLAMA_CPP_LIB_PATH" not in kwargs["env"]
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        direct = "-S" in argv
        if self.fail == ("direct" if direct else "python"):
            return subprocess.CompletedProcess(argv, 1, "", "mock failure")
        path = Path(argv[-1])
        if direct:
            result = {"libllama": str(path / "libllama.so"),
                      "cuda_supported": self.direct_gpu,
                      "backend_initialized": True}
        else:
            result = {"version": "0.3.36", "cuda_supported": self.python_gpu,
                      "same_libllama": self.same,
                      "actual_library": str(path if self.same else path.parent / "foreign.so")}
        return subprocess.CompletedProcess(argv, 0, json.dumps(result) + "\n", "")


def verify(root, env, run=None):
    return verify_native_library(
        project_root=root, environment=env, run=run or SubprocessProbe())


class NativeLibraryQualificationTests(unittest.TestCase):
    def test_full_dual_abi_qualification_no_model_inference(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, _ = fixture(root)
            execution = SubprocessProbe()
            result = verify(root, env, execution)
            self.assertEqual(result.status, "native_library_dual_abi_verified")
            self.assertTrue(result.shared_libraries_verified)
            self.assertTrue(result.python_binding_same_library)
            self.assertTrue(result.native_backend_initialized)
            self.assertTrue(result.cuda_support_direct)
            self.assertTrue(result.cuda_support_python)
            self.assertFalse(result.gpu_offload_executed)
            self.assertFalse(result.inference_executed)
            self.assertFalse(result.compilation_started)
            self.assertFalse(result.installation_started)
            self.assertFalse(result.cpu_fallback_selected)
            self.assertEqual(len(execution.calls), 2)
            self.assertIn("-S", execution.calls[0][0])
            self.assertNotIn("-S", execution.calls[1][0])

    def test_modified_installed_lib_is_rejected_prior_to_loading(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, lib = fixture(root)
            (lib / "libllama.so").write_bytes(b"tampered")
            def forbidden(*a, **k):
                self.fail("Do not execute modified binary")
            result = verify_native_library(
                project_root=root, environment=env, run=forbidden)
            self.assertEqual(result.status, "native_library_integrity_failed")

    def test_modified_cuda_backend_is_rejected_prior_to_loading(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, lib = fixture(root)
            (lib / "libggml-cuda.so").write_bytes(b"modified")
            self.assertEqual(verify(root, env).status,
                             "native_library_integrity_failed")

    def test_modified_activation_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, _ = fixture(root)
            activation = root / ".venv/lib/python3.12/site-packages" / PTH_FILE
            activation.write_text("/usr/local/lib\n")
            self.assertEqual(verify(root, env).status,
                             "native_library_integrity_failed")

    def test_symlinked_library_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, lib = fixture(root)
            target = lib / "libllama.so"
            payload = target.read_bytes()
            target.unlink()
            outside = root / "outside"
            outside.write_bytes(payload)
            target.symlink_to(outside)
            self.assertEqual(verify(root, env).status,
                             "native_library_integrity_failed")

    def test_pinned_wheel_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, _ = fixture(root)
            lock = root / NATIVE_LOCK
            doc = json.loads(lock.read_text())
            doc["wheel_sha256"] = "0" * 64
            lock.write_text(json.dumps(doc))
            self.assertEqual(verify(root, env).status,
                             "native_library_integrity_failed")

    def test_direct_gpu_unavailable_blocks_python_probe(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, _ = fixture(root)
            probes = SubprocessProbe(direct_gpu=False)
            result = verify(root, env, probes)
            self.assertEqual(result.status, "native_direct_cuda_unavailable")
            self.assertEqual(len(probes.calls), 1)

    def test_python_binding_uses_different_library_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, _ = fixture(root)
            result = verify(root, env, SubprocessProbe(same=False))
            self.assertEqual(result.status, "native_python_library_mismatch")

    def test_python_binding_cannot_run_cpu_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, _ = fixture(root)
            result = verify(root, env, SubprocessProbe(python_gpu=False))
            self.assertEqual(result.status, "native_python_library_mismatch")

    def test_native_subprocess_exception_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, _ = fixture(root)
            result = verify(root, env, SubprocessProbe(fail="direct"))
            self.assertEqual(result.status, "native_library_probe_failed")

    def test_invalid_venv_prevents_native_library_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, _ = fixture(root)
            env.interpreter = "/usr/bin/python3"
            result = verify(root, env)
            self.assertEqual(result.status, "blocked_venv")

    def test_missing_installed_library_is_not_automatically_recompiled(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, lib = fixture(root)
            (lib / "libggml-cuda.so").unlink()
            result = verify(root, env)
            self.assertEqual(result.status, "native_library_integrity_failed")
            self.assertFalse(result.compilation_started)


class FakeFunction:
    def __init__(self, result=None):
        self.result = result
        self.calls = 0
        self.argtypes = None
        self.restype = None

    def __call__(self):
        self.calls += 1
        return self.result


class FakeLibrary:
    def __init__(self, gpu=True):
        self.llama_supports_gpu_offload = FakeFunction(gpu)
        self.llama_backend_init = FakeFunction()
        self.llama_backend_free = FakeFunction()
        self.llama_print_system_info = FakeFunction(b"mock cuda native backend")


class DirectNativeAdapterTests(unittest.TestCase):
    def test_manages_native_abi_lifetime_without_models(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, lib = fixture(root)
            loaded = FakeLibrary()
            from ruth_runtime.native_library import _report
            trusted = _report("native_library_dual_abi_verified",
                              libs=True, python_same=True,
                              gpu_direct=True, gpu_python=True)
            adapter = DirectLlamaLibraryAdapter(
                project_root=root, verify=lambda **kwargs: trusted,
                load_library=lambda path: loaded,
            )
            with adapter.open() as native:
                self.assertEqual(native.system_info(), "mock cuda native backend")
                self.assertEqual(loaded.llama_backend_init.calls, 1)
                self.assertEqual(loaded.llama_backend_free.calls, 0)
            self.assertEqual(loaded.llama_backend_free.calls, 1)
            native.close()
            self.assertEqual(loaded.llama_backend_free.calls, 1)
            with self.assertRaises(DirectNativeUnavailable):
                native.system_info()

    def test_fails_closed_before_loading_unqualified_abi(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, lib = fixture(root)
            from ruth_runtime.native_library import _report
            adapter = DirectLlamaLibraryAdapter(
                project_root=root,
                verify=lambda **kwargs: _report("native_library_integrity_failed"),
                load_library=lambda path: self.fail("should not load"),
            )
            with self.assertRaises(DirectNativeUnavailable):
                adapter.open()

    def test_rejects_native_library_with_gpu_offload_unavailable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, lib = fixture(root)
            from ruth_runtime.native_library import _report
            trusted = _report("native_library_dual_abi_verified",
                              libs=True, python_same=True,
                              gpu_direct=True, gpu_python=True)
            native = FakeLibrary(gpu=False)
            adapter = DirectLlamaLibraryAdapter(
                project_root=root, verify=lambda **kwargs: trusted,
                load_library=lambda path: native,
            )
            with self.assertRaises(DirectNativeUnavailable):
                adapter.open()
            self.assertEqual(native.llama_backend_init.calls, 0)


if __name__ == "__main__":
    unittest.main()
