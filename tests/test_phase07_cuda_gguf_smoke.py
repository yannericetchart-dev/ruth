"""GPU GGUF probe contract tests using only mock subprocesses and tiny files."""
from dataclasses import replace
import hashlib
import json
from io import StringIO
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruth_runtime.cuda_adapter import (
    CudaLlamaAdapter, INITIAL_STACK, UnverifiedCudaEngine,
)
from ruth_runtime.inference_probe import (
    _parse_offload, _valid_local_gguf, probe_inference,
    WORKER_PREFIX, WORKER_SCRIPT,
)
from ruth_runtime.inference_worker import run_worker
from ruth_runtime.launcher import main
from ruth_runtime.model_catalog import select_model
from ruth_runtime.model_download import _local_path, _manifest

DATA = b"GGUF" + b"RUTHFAKE" * 20
UUID = "GPU-0000aabb-1111-2222-3333-ccddeeff0011"
PCI = "00000000:01:00.0"


def fixture(root):
    original = select_model("test")
    choice = replace(original, expected_size_bytes=len(DATA),
                     expected_sha256=hashlib.sha256(DATA).hexdigest())
    path = _local_path(root, choice)
    path.parent.mkdir(parents=True)
    path.write_bytes(DATA)
    manifest = {
        **_manifest(choice), "verified": True,
        "license_acknowledged_for_this_download": True,
    }
    path.with_name(path.name + ".manifest.json").write_text(json.dumps(manifest))
    env = SimpleNamespace(
        interpreter_in_project_venv=True,
        python_version="3.12.3", project_root=str(root),
        interpreter=str(root / ".venv/bin/python"),
    )
    return choice, path, env


def trusted_native(**kwargs):
    return SimpleNamespace(status="native_library_dual_abi_verified")


class FakeWorker:
    def __init__(self, *, tokens=10, loaded=True, observed=True,
                 peak=948, offloaded=True, success=True, worker_uuid=UUID):
        self.tokens = tokens
        self.loaded = loaded
        self.observed = observed
        self.peak = peak
        self.offloaded = offloaded
        self.success = success
        self.worker_uuid = worker_uuid
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        assert argv[:3] == [
            str(Path(kwargs["cwd"]) / ".venv/bin/python"), "-I", "-c",
        ]
        assert argv[3] == WORKER_SCRIPT
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == UUID
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        assert "LLAMA_CPP_LIB_PATH" not in kwargs["env"]
        assert kwargs["timeout"] == 240
        output = {
            "schema": "ruth-cuda-gguf-smoke-v1",
            "purpose": "test", "gpu_uuid": self.worker_uuid,
            "model_loaded": self.loaded,
            "completion_tokens": self.tokens,
            "gpu_process_observed": self.observed,
            "gpu_memory_peak_mib_observed": self.peak,
            "worker_duration_seconds": 1.2,
        }
        log = "llama_model_load_tensors: offloaded 17/17 layers to GPU" \
            if self.offloaded else "llama_model_load_tensors: CPU buffer only"
        return subprocess.CompletedProcess(
            argv, 0 if self.success else 2,
            WORKER_PREFIX + json.dumps(output) + "\n", log,
        )


class GpuInferenceContractTests(unittest.TestCase):
    def verify(self, root, *, fake=None, purpose="test", allow=True,
               choice=None, native=trusted_native, discover=None):
        selected, path, env = fixture(root) if choice is None else choice
        worker = fake or FakeWorker()
        with patch("ruth_runtime.inference_probe.select_model",
                   return_value=selected):
            result = probe_inference(
                project_root=root, purpose=purpose,
                authorize_gpu_inference=allow,
                environment=env, run=worker,
                native_verification=native,
                gpu_discovery=discover or (lambda **kwargs: (UUID, PCI)),
            )
        return result, worker

    def test_no_consent_does_not_inspect_hardware_or_disk(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = probe_inference(
                project_root=root, authorize_gpu_inference=False,
                environment=None,
                run=lambda *a, **kw: self.fail("No subprocess"),
                native_verification=lambda **kw: self.fail("No native probe"),
            )
            self.assertEqual(result.status, "gpu_inference_not_authorized")
            self.assertFalse(result.model_loaded)
            self.assertFalse(result.gpu_offload_executed)

    def test_full_real_evidence_contract_only_verified_in_mock(self):
        with TemporaryDirectory() as tmp:
            result, worker = self.verify(Path(tmp))
            self.assertEqual(result.status, "gpu_gguf_inference_verified")
            self.assertTrue(result.model_verified)
            self.assertTrue(result.native_library_verified)
            self.assertEqual(result.completion_tokens, 10)
            self.assertEqual(result.offloaded_layers, 17)
            self.assertEqual(result.gpu_memory_peak_mib_observed, 948)
            self.assertTrue(result.gpu_inference_verified)
            self.assertFalse(result.cpu_fallback_selected)
            self.assertEqual(len(worker.calls), 1)

    def test_model_checksum_mismatch_prevents_native_process(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            choice, path, env = fixture(root)
            path.write_bytes(b"GGUFtampered")
            with patch("ruth_runtime.inference_probe.select_model", return_value=choice):
                result = probe_inference(
                    project_root=root, authorize_gpu_inference=True,
                    environment=env,
                    native_verification=lambda **kw: self.fail("Do not load native"),
                )
            self.assertEqual(result.status, "model_integrity_failed")

    def test_model_sidecar_without_license_consent_refused(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            choice, path, env = fixture(root)
            manifest = path.with_name(path.name + ".manifest.json")
            data = json.loads(manifest.read_text())
            data["license_acknowledged_for_this_download"] = False
            manifest.write_text(json.dumps(data))
            self.assertFalse(_valid_local_gguf(root, choice)[0])

    def test_no_offloaded_layers_must_be_inconclusive(self):
        with TemporaryDirectory() as tmp:
            result, _ = self.verify(Path(tmp), fake=FakeWorker(offloaded=False))
            self.assertEqual(result.status, "gpu_inference_inconclusive")
            self.assertFalse(result.gpu_offload_executed)
            self.assertTrue(result.inference_executed)
            self.assertFalse(result.gpu_inference_verified)

    def test_no_gpu_pid_memory_must_be_inconclusive(self):
        with TemporaryDirectory() as tmp:
            result, _ = self.verify(Path(tmp), fake=FakeWorker(observed=False, peak=None))
            self.assertEqual(result.status, "gpu_inference_inconclusive")
            self.assertFalse(result.gpu_inference_verified)

    def test_no_completion_tokens_must_be_inconclusive(self):
        with TemporaryDirectory() as tmp:
            result, _ = self.verify(Path(tmp), fake=FakeWorker(tokens=0))
            self.assertFalse(result.inference_executed)
            self.assertFalse(result.gpu_inference_verified)

    def test_worker_failure_not_promoted_as_success(self):
        with TemporaryDirectory() as tmp:
            result, _ = self.verify(Path(tmp), fake=FakeWorker(success=False))
            self.assertEqual(result.status, "gpu_inference_worker_failed")
            self.assertFalse(result.gpu_inference_verified)

    def test_different_gpu_uuid_fails_closed(self):
        with TemporaryDirectory() as tmp:
            result, _ = self.verify(Path(tmp), fake=FakeWorker(worker_uuid="GPU-other"))
            self.assertEqual(result.status, "gpu_inference_probe_failed")

    def test_production_inference_is_blocked(self):
        with TemporaryDirectory() as tmp:
            result = probe_inference(
                project_root=Path(tmp), purpose="production",
                authorize_gpu_inference=True, run=lambda *a, **kw: self.fail("no run"),
            )
            self.assertEqual(result.status, "production_inference_not_authorized")

    def test_native_library_validation_required(self):
        with TemporaryDirectory() as tmp:
            result, worker = self.verify(
                Path(tmp), native=lambda **kw: SimpleNamespace(status="native_unverified"))
            self.assertEqual(result.status, "native_runtime_unverified")
            self.assertEqual(worker.calls, [])

    def test_worker_exception_does_not_claim_gpu_inference(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            class Failing:
                def __call__(self, argv, **kwargs):
                    raise subprocess.TimeoutExpired(argv, 240)
            result, _ = self.verify(root, fake=Failing())
            self.assertEqual(result.status, "gpu_inference_probe_failed")
            self.assertFalse(result.gpu_inference_verified)

    def test_parser_accepts_only_actual_layer_offload_log(self):
        self.assertEqual(_parse_offload("offloaded 17/17 layers to GPU"), (17, 17))
        self.assertEqual(_parse_offload("offloaded 0/17 layers to GPU"), (0, 17))
        self.assertEqual(_parse_offload("CUDA supported"), (0, 0))


class WorkerTests(unittest.TestCase):
    def test_worker_validates_selected_gpu_and_generates_tokens(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected, file, env = fixture(root)
            class FakeSession:
                def __init__(self):
                    self.closed = False
                def generate_report(self, prompt, *, max_tokens):
                    self.assertion = (prompt, max_tokens)
                    return {"completion_tokens": 5, "text": "Ciao"}
                def close(self):
                    self.closed = True
            session = FakeSession()
            with patch("ruth_runtime.inference_worker.select_model",
                       return_value=selected), \
                 patch("ruth_runtime.inference_worker.CudaLlamaAdapter") as adapter, \
                 patch("ruth_runtime.inference_worker.measure_worker_gpu",
                       side_effect=lambda pid, uuid, stop, values: values.append(93)), \
                 patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": UUID}):
                adapter.return_value.load.return_value = session
                result = run_worker(root, "test", UUID)
                self.assertTrue(adapter.call_args.kwargs["native_diagnostics"])
                self.assertEqual(adapter.return_value.load.call_args.args[1],
                                 INITIAL_STACK)
            self.assertTrue(session.closed)
            self.assertTrue(result["model_loaded"])
            self.assertEqual(result["completion_tokens"], 5)
            self.assertEqual(result["gpu_memory_peak_mib_observed"], 93)
            self.assertEqual(result["generated_text_sha256"],
                             hashlib.sha256(b"Ciao").hexdigest())

    def test_wrong_gpu_binding_fails_without_loading_backend(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected, _, _ = fixture(root)
            with patch("ruth_runtime.inference_worker.select_model",
                       return_value=selected), \
                 patch("ruth_runtime.inference_worker.CudaLlamaAdapter",
                       side_effect=AssertionError("No native import")), \
                 patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-wrong"}):
                with self.assertRaises(RuntimeError):
                    run_worker(root, "test", UUID)

    def test_completion_accounting_rejects_missing_usage(self):
        from ruth_runtime.cuda_adapter import CudaLlamaAdapter
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            gguf = root / "test.gguf"
            gguf.write_bytes(b"mock")
            class FakeLlama:
                cache = None
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                def create_completion(self, **kwargs):
                    return {"choices":[{"text":"Ciao"}], "usage": {"completion_tokens": 3}}
                def close(self): pass
            backend = SimpleNamespace(
                __version__="0.3.36",
                llama_supports_gpu_offload=lambda: True, Llama=FakeLlama,
            )
            adapter = CudaLlamaAdapter(
                load_backend=lambda _: backend, native_diagnostics=True)
            session = adapter.load(gguf, INITIAL_STACK, {})
            self.assertEqual(session.generate_report("ciao")["completion_tokens"], 3)
            self.assertTrue(session.cache_disabled)
            self.assertTrue(session._llm.kwargs["verbose"])
            session.close()


class GpuSmokeCliTests(unittest.TestCase):
    def test_cli_no_gpu_inference_without_explicit_flag(self):
        from types import SimpleNamespace
        report = SimpleNamespace(
            gpu_inference_verified=False,
            to_dict=lambda: {"status": "gpu_inference_not_authorized"},
        )
        with patch("ruth_runtime.inference_probe.probe_inference",
                   return_value=report) as probe, \
             patch("sys.stdout", new_callable=StringIO) as stdout:
            self.assertEqual(main(["models", "smoke", "--purpose", "test"]), 2)
            self.assertFalse(probe.call_args.kwargs["authorize_gpu_inference"])
            self.assertIn("gpu_inference_not_authorized", stdout.getvalue())

    def test_cli_gpu_inference_permission_explicit_and_bound_to_test_model(self):
        report = SimpleNamespace(
            gpu_inference_verified=True,
            to_dict=lambda: {"status": "gpu_gguf_inference_verified"},
        )
        with patch("ruth_runtime.inference_probe.probe_inference",
                   return_value=report) as probe, \
             patch("sys.stdout", new_callable=StringIO) as stdout:
            self.assertEqual(main(["models", "smoke", "--purpose", "test",
                                   "--allow-gpu-inference"]), 0)
            self.assertTrue(probe.call_args.kwargs["authorize_gpu_inference"])
            self.assertEqual(probe.call_args.kwargs["purpose"], "test")
            self.assertIn("gpu_gguf_inference_verified", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
