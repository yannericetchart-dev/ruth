"""Offline load-only regression coverage: no real GGUF, CUDA, or generation."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruth_runtime.load_only_probe import (
    PREFIX, SCHEMA, extract_native_buffers, probe_model_load,
)
from ruth_runtime.launcher import main


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
            "inference_executed": False, "generated_tokens": 0,
            "gpu_process_observed": self.loaded,
            "gpu_memory_peak_mib_observed": 5450 if self.loaded else None,
            "cuda_free_mib_after_backend_init": 5700,
            "cuda_free_mib_after_model_load": 300 if self.loaded else None,
            "cuda_free_mib_after_context_init": 250 if self.context else None,
            "error": self.error,
            "n_ctx": 128 if self.mode == "context" else None,
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


class LoadOnlyTests(unittest.TestCase):
    def execute(self, mode="weights", worker=None, preflight=ready):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
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
                    allow_gpu_load=True, mode=mode, environment=env,
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
