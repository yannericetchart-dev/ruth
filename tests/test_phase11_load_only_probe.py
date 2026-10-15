"""Offline load-only regression coverage: no real GGUF, CUDA, or generation."""
import json
import shutil
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruth_runtime.load_only_probe import (
    PREFIX, SCHEMA, extract_native_buffers, probe_model_load,
)
from ruth_runtime.launcher import main
from ruth_runtime.load_only_worker import run_load_only
from ruth_runtime.install_plan import WHEEL_VERSION
from ruth_runtime.dependencies import PROJECT_ROOT


UUID = "GPU-5a841828-882b-85d9-c98e-3a0d29e66e00"
PCI = "00000000:01:00.0"


class FakeWorker:
    def __init__(self, *, mode="weights", returncode=0, loaded=True,
                 context=False, error=None):
        self.calls = []
        self.mode, self.returncode = mode, returncode
        self.loaded, self.context, self.error = loaded, context, error

    def __call__(self, args, **kw):
        self.calls.append((args, kw))
        self_guard = kw["env"]["CUDA_VISIBLE_DEVICES"] == UUID
        if not self_guard:
            raise AssertionError("Wrong GPU binding")
        if "--allow-gpu-inference" in args:
            raise AssertionError("No inference permitted")
        payload = {
            "schema": SCHEMA, "mode": self.mode,
            "quantization": "Q4_K_L", "gpu_uuid": UUID,
            "weights_loaded": self.loaded, "context_initialized": self.context,
            "cuda_preinit_requested": args[-1] == "preinit",
            "cuda_primary_context_preinitialized": args[-1] == "preinit",
            "cuda_free_mib_after_primary_ctx_init": (
                5680 if args[-1] == "preinit" else None),
            "inference_executed": False, "generated_tokens": 0,
            "gpu_process_observed": self.loaded,
            "gpu_memory_peak_mib_observed": 5450 if self.loaded else None,
            "cuda_free_mib_after_backend_init": 5700,
            "cuda_free_mib_after_model_load": 300 if self.loaded else None,
            "cuda_free_mib_after_context_init": 250 if self.context else None,
            "error": self.error,
            "n_ctx": int(args[-2]) if self.mode == "context" else None,
            "n_batch": 64 if self.mode == "context" else None,
            "n_ubatch": 64 if self.mode == "context" else None,
        }
        stderr = ("llama_model_load_tensors: offloaded 43/43 layers to GPU\n"
                  "llama_model_load_tensors: CUDA0 model buffer size = 5700.31 MiB\n")
        if not self.loaded:
            stderr = ("ggml_backend_cuda_buffer_type_alloc_buffer: allocating "
                      "5700.31 MiB on device 0: cudaMalloc failed: out of memory\n")
        return subprocess.CompletedProcess(
            args, self.returncode,
            PREFIX + json.dumps(payload) + "\n", stderr,
        )


def ready(uuid, **kw):
    return SimpleNamespace(
        status="gpu_ready", error=None,
        to_dict=lambda: {
            "status": "gpu_ready", "gpu_uuid": uuid,
            "processes": [], "memory_mib": {
                "total": 6141, "reserved": 369, "used": 2, "free": 5772,
            },
        },
    )



class LoadOnlyWorkerNoInferenceTests(unittest.TestCase):
    def test_primary_context_preinitializes_before_native_model_load(self):
        calls = []
        class Driver:
            def cuCtxSetCurrent(self, ctx):
                calls.append("clear_context")
            def cuDevicePrimaryCtxRelease(self, device):
                calls.append("release_context")

        class API:
            def llama_supports_gpu_offload(self): return True
            def llama_backend_init(self): calls.append("backend_init")
            def llama_backend_free(self): calls.append("backend_free")
            def llama_model_default_params(self):
                return SimpleNamespace(n_gpu_layers=0, main_gpu=0, use_mmap=False)
            def llama_model_load_from_file(self, path, params):
                calls.append("model_load")
                return None

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.gguf"
            model.write_bytes(b"GGUF01234567")
            choice = SimpleNamespace(
                family="gemma", model="gemma-2",
                version="gemma-2-9B-it", quantization="Q4_K_L",
                expected_size_bytes=model.stat().st_size,
            )
            package = ModuleType("llama_cpp")
            package.__version__ = WHEEL_VERSION
            package.llama_cpp = API()
            def initialize():
                calls.append("preinit")
                return Driver(), 0
            with patch.dict(sys.modules, {"llama_cpp": package}), \
                 patch("ruth_runtime.load_only_worker.select_model",
                       return_value=choice), \
                 patch("ruth_runtime.load_only_worker._local_path",
                       return_value=model), \
                 patch("ruth_runtime.load_only_worker.measure_worker_gpu"), \
                 patch("ruth_runtime.load_only_worker.initialize_cuda_primary_context",
                       side_effect=initialize), \
                 patch("ruth_runtime.load_only_worker.current_cuda_free_mib",
                       return_value=5682.8), \
                 patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": UUID}):
                result = run_load_only(root, UUID, "Q4_K_L", "weights",
                                       128, cuda_preinit=True)
            self.assertEqual(calls[:3], ["backend_init", "preinit", "model_load"])
            self.assertIn("release_context", calls)
            self.assertTrue(result["cuda_primary_context_preinitialized"])
            self.assertEqual(result["cuda_free_mib_after_primary_ctx_init"], 5682.8)
            self.assertFalse(result["weights_loaded"])
            self.assertFalse(result["inference_executed"])

    def test_native_stage_separation_does_not_generate_any_tokens(self):
        class FakeAPI:
            def __init__(self):
                self.calls = []
            def llama_supports_gpu_offload(self):
                return True
            def llama_backend_init(self):
                self.calls.append("backend_init")
            def llama_backend_free(self):
                self.calls.append("backend_free")
            def llama_model_default_params(self):
                return SimpleNamespace(n_gpu_layers=0, main_gpu=0, use_mmap=False)
            def llama_model_load_from_file(self, path, params):
                self.calls.append(("model_load", params.n_gpu_layers))
                return object()
            def llama_model_free(self, model):
                self.calls.append("model_free")
            def llama_context_default_params(self):
                return SimpleNamespace(n_ctx=0, n_batch=0, n_ubatch=0,
                                       offload_kqv=False, swa_full=False,
                                       n_seq_max=0, kv_unified=True,
                                       flash_attn_type=0, type_k=0, type_v=0)
            def llama_init_from_model(self, model, params):
                self.calls.append(("context_init", params.n_ctx))
                return object()
            GGML_TYPE_F16 = 1
            LLAMA_FLASH_ATTN_TYPE_DISABLED = 0
            LLAMA_FLASH_ATTN_TYPE_AUTO = -1
            LLAMA_FLASH_ATTN_TYPE_ENABLED = 1
            def llama_free(self, context):
                self.calls.append("context_free")

        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "model.gguf"
            path.write_bytes(b"GGUF123456")
            choice = SimpleNamespace(
                family="gemma", model="gemma-2",
                version="gemma-2-9B-it", quantization="Q4_K_L",
                expected_size_bytes=path.stat().st_size,
            )
            for mode in ("weights", "context"):
                api = FakeAPI()
                package = ModuleType("llama_cpp")
                package.__version__ = WHEEL_VERSION
                package.llama_cpp = api
                with patch.dict(sys.modules, {"llama_cpp": package}), \
                     patch("ruth_runtime.load_only_worker.select_model",
                           return_value=choice), \
                     patch("ruth_runtime.load_only_worker._local_path",
                           return_value=path), \
                     patch("ruth_runtime.load_only_worker.measure_worker_gpu"), \
                     patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": UUID}):
                    result = run_load_only(root, UUID, "Q4_K_L", mode, 257)
                self.assertTrue(result["weights_loaded"])
                self.assertEqual(result["generated_tokens"], 0)
                self.assertFalse(result["inference_executed"])
                self.assertEqual(result["context_initialized"], mode == "context")
                self.assertTrue(any(isinstance(x, tuple) and x[0] == "model_load"
                                    and x[1] == 2**31 - 1 for x in api.calls))
                self.assertEqual(any(isinstance(x, tuple) and
                                     x[0] == "context_init" for x in api.calls),
                                 mode == "context")


class LoadOnlyTests(unittest.TestCase):
    def execute(self, mode="weights", worker=None, preflight=ready,
                cuda_preinit=False):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "catalog").mkdir()
            shutil.copyfile(
                PROJECT_ROOT / "catalog/llama_kv_backend_policy.json",
                root / "catalog/llama_kv_backend_policy.json")
            env = SimpleNamespace(
                interpreter_in_project_venv=True,
                python_version="3.12.3", project_root=str(root),
                interpreter=str(root / ".venv/bin/python"),
            )
            proc = worker or FakeWorker(mode=mode, context=(mode == "context"))
            with patch("ruth_runtime.load_only_probe.select_model",
                       return_value=SimpleNamespace(
                           family="gemma", quantization="Q4_K_L")), \
                 patch("ruth_runtime.load_only_probe._valid_local_gguf",
                       return_value=(True, root / "fake.gguf", None)):
                report = probe_model_load(
                    project_root=root, quantization="Q4_K_L",
                    allow_gpu_load=True, mode=mode,
                    cuda_preinit=cuda_preinit, environment=env,
                    verify_native=lambda **kw: SimpleNamespace(
                        status="native_library_dual_abi_verified",
                    ),
                    discover_gpu=lambda **kw: (UUID, PCI),
                    gpu_preflight=preflight, run=proc,
                )
            return report, proc

    def test_native_buffer_parser(self):
        rows = extract_native_buffers(
            "llama_model_load_tensors: CUDA0 model buffer size = 5700.31 MiB\n"
            "llama_kv_cache_init: CUDA0 KV buffer size = 12.00 MiB\n"
            "llama_init_from_model: CUDA0 compute buffer size = 33.50 MiB"
        )
        self.assertEqual([row["kind"] for row in rows],
                         ["model", "kv", "compute"])
        self.assertEqual(rows[0]["mib"], 5700.31)

    def test_cuda_preinit_reported_separately_from_original_baseline(self):
        result, worker = self.execute(cuda_preinit=True)
        self.assertEqual(result["status"], "gpu_load_only_completed")
        self.assertTrue(result["cuda_preinit_requested"])
        self.assertTrue(result["cuda_primary_context_preinitialized"])
        self.assertEqual(result["cuda_free_mib_after_primary_ctx_init"], 5680)
        self.assertEqual(worker.calls[0][0][-1], "preinit")
        self.assertFalse(result["inference_executed"])

    def test_weights_only_no_inference(self):
        result, worker = self.execute()
        self.assertEqual(result["status"], "gpu_load_only_completed")
        self.assertTrue(result["weights_loaded"])
        self.assertFalse(result["context_initialized"])
        self.assertFalse(result["inference_executed"])
        self.assertEqual(result["generated_tokens"], 0)
        self.assertEqual(len(worker.calls), 1)
        self.assertTrue(result["full_gpu_offload_observed"])
        self.assertFalse(result["runtime_profile_published"])

    def test_native_kv_padding_observed_not_inferred(self):
        result, worker = self.execute(mode="context")
        self.assertEqual(result["n_ctx"], 128)
        self.assertEqual(result["native_kv_sizes"], [])
        self.assertFalse(result["inference_executed"])

    def test_context_buffers_without_generation(self):
        result, _ = self.execute(mode="context")
        self.assertEqual(result["status"], "gpu_load_only_completed")
        self.assertTrue(result["context_initialized"])
        self.assertFalse(result["inference_executed"])
        self.assertEqual(result["cuda_free_mib_after_context_init"], 250)

    def test_cuda_oom_is_not_mislabeled_or_promoted(self):
        result, _ = self.execute(worker=FakeWorker(
            returncode=2, loaded=False, error="CUDA out of memory"
        ))
        self.assertEqual(result["status"], "gpu_load_only_failed")
        self.assertEqual(result["native_failure_category"],
                         "cuda_out_of_memory_reported")
        self.assertEqual(result["native_cuda_buffer_mib_reported"], 5700.31)
        self.assertFalse(result["runtime_profile_published"])

    def test_gpu_busy_refuses_worker(self):
        def busy(uuid, **kw):
            return SimpleNamespace(
                status="gpu_busy", error=None,
                to_dict=lambda: {"status": "gpu_busy",
                                 "processes": [{"pid": 3032,
                                                "name": "Xorg"}]},
            )
        result, worker = self.execute(preflight=busy)
        self.assertEqual(result["status"], "gpu_busy")
        self.assertEqual(worker.calls, [])

    def test_flag_required_before_any_disk_or_gpu(self):
        result = probe_model_load(allow_gpu_load=False)
        self.assertEqual(result["status"], "gpu_load_not_authorized")
        self.assertFalse(result["inference_executed"])

    def test_cli_does_not_authorize_inference(self):
        with patch("ruth_runtime.load_only_probe.probe_model_load") as probe:
            probe.return_value = {
                "status": "gpu_load_not_authorized",
                "inference_executed": False,
            }
            self.assertEqual(main(["models", "probe-load"]), 2)
            self.assertFalse(probe.call_args.kwargs["allow_gpu_load"])
            self.assertEqual(probe.call_args.kwargs["mode"], "weights")


if __name__ == "__main__":
    unittest.main()
