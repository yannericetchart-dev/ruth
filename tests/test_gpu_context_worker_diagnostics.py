"""CPU-only regression tests for Qwen's inconclusive GPU context worker."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from calibration.gpu_context import probe_context, search_gpu_context


class ContextWorkerDiagnosticsTests(unittest.TestCase):
    def test_signal_killed_worker_with_no_json_is_inconclusive(self):
        layout = {"tensors": [], "metadata": {"general.architecture": "qwen2"}}
        native_confirmation = {
            "verified": False, "tensor_count": 0,
            "unverified_names": [],
            "native_override": {"native_logged_occurrence_count": 0},
        }
        with (
            patch("calibration.gpu_context.inspect_gguf",
                  return_value=layout),
            patch("calibration.gpu_context.subprocess.run",
                  return_value=subprocess.CompletedProcess(
                      [], -9, "", ""
                  )),
            patch("calibration.gpu_context.parse_native_memory",
                  return_value={"native_buffer_records": []}),
            patch("calibration.gpu_context.native_confirmation",
                  return_value=native_confirmation),
        ):
            report = probe_context(Path("/test/model.gguf"), 2560)
        self.assertFalse(report["success"])
        self.assertIs(report["worker_json_valid"], False)
        self.assertEqual(report["worker_signal"], "SIGKILL")
        self.assertEqual(report["worker_returncode"], -9)
        self.assertEqual(report["worker_stdout_chars"], 0)
        self.assertEqual(report["failure_class"], "worker_protocol_error")
        self.assertFalse(report["suspected_cuda_oom"])
        self.assertIn("non assumere OOM CUDA", report["error"])

    def test_non_object_json_is_rejected_without_type_error(self):
        layout = {"tensors": [], "metadata": {"general.architecture": "qwen2"}}
        native_confirmation = {
            "verified": False, "tensor_count": 0,
            "unverified_names": [],
            "native_override": {"native_logged_occurrence_count": 0},
        }
        for stdout in ("[]", "null", "42", "true", '"hello"'):
            with (
                self.subTest(stdout=stdout),
                patch("calibration.gpu_context.inspect_gguf",
                      return_value=layout),
                patch("calibration.gpu_context.subprocess.run",
                      return_value=subprocess.CompletedProcess(
                          [], 0, stdout, ""
                      )),
                patch("calibration.gpu_context.parse_native_memory",
                      return_value={"native_buffer_records": []}),
                patch("calibration.gpu_context.native_confirmation",
                      return_value=native_confirmation),
            ):
                report = probe_context(Path("/test/model.gguf"), 2560)
                self.assertFalse(report["success"])
                self.assertIs(report["worker_json_valid"], False)
                self.assertEqual(report["failure_class"], "worker_protocol_error")
                self.assertIn("malformato/non oggetto", report["error"])

    def test_cublas_abort_has_native_boundary_not_json_protocol_cause(self):
        layout = {"tensors": [], "metadata": {"general.architecture": "qwen2"}}
        placement = {
            "verified": True, "tensor_count": 339,
            "unverified_names": [],
            "native_override": {"native_logged_occurrence_count": 339},
        }
        stderr = (
            "llama_kv_cache: CUDA0 KV buffer size = 140.00 MiB\n"
            "CUDA error: the resource allocation failed\n"
            "current device: 0, in function cublas_handle\n"
            "cublasCreate_v2(&cublas_handles[device][curr_stream_no])\n"
        )
        with (
            patch("calibration.gpu_context.inspect_gguf",
                  return_value=layout),
            patch("calibration.gpu_context.subprocess.run",
                  return_value=subprocess.CompletedProcess(
                      [], -6, "", stderr
                  )),
            patch("calibration.gpu_context.parse_native_memory",
                  return_value={"native_buffer_records": []}),
            patch("calibration.gpu_context.native_confirmation",
                  return_value=placement),
        ):
            report = probe_context(Path("/test/model.gguf"), 2560)
        self.assertFalse(report["success"])
        self.assertEqual(report["worker_signal"], "SIGABRT")
        self.assertIs(report["worker_json_valid"], False)
        self.assertEqual(
            report["failure_class"], "native_cuda_resource_allocation_failure"
        )
        self.assertFalse(report["suspected_cuda_oom"])
        self.assertTrue(report["native_cuda_resource_allocation_failed"])

    @staticmethod
    def gpu():
        return {
            "ready": True, "errors": [],
            "gpu": {
                "name": "Mock NVIDIA", "vendor": "nvidia",
                "uuid": "GPU-test", "index": 0,
            },
            "memory": {"total_mib": 6141},
        }

    @staticmethod
    def success(ctx):
        return {
            "success": True, "n_ctx": ctx, "failure_class": None,
            "measurements": {"vram_peak_sampled_mib": 6100.0},
            "full_gpu_native_verification": {"verified": True},
            "gpu_runtime_validation": {
                "verified": True, "kv": {"verified_gpu_resident": True},
            },
        }

    def run_fake_search(self, fake_probe):
        from model import ModelManager
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "Qwen2.5-7B-Instruct-Q5_K_L.gguf"
            path.write_bytes(b"mock")
            baseline = {
                "native_verified_all_gpu": True,
                "model_id": "qwen-2.5-7b-instruct",
                "source": "bartowski",
                "repository": ModelManager.source_spec(
                    "qwen-2.5-7b-instruct", "bartowski"
                )["repository"],
                "filename": path.name,
                "local_path": str(path),
                "architecture": "qwen2",
                "quantization": "Q5_K_L",
            }
            metadata = {
                "general.architecture": "qwen2",
                "qwen2.context_length": 129,
            }
            events = []
            with (
                patch("calibration.gpu_context.GPUManager.calibration_status",
                      return_value=self.gpu()),
                patch("calibration.gpu_context.inspect_gguf",
                      return_value={"metadata": metadata}),
                patch("calibration.gpu_context.probe_context",
                      side_effect=fake_probe),
            ):
                result = search_gpu_context(
                    "qwen-2.5-7b-instruct",
                    root=folder, baseline_selection=baseline,
                    source="bartowski", max_ctx=129, min_ctx=64,
                    persist_reference=False,
                    progress=events.append,
                )
            return result, events

    def test_one_protocol_retry_then_success_is_confirmed(self):
        calls = []
        def fake_probe(path, ctx, **kwargs):
            calls.append(ctx)
            if ctx == 129 and calls.count(129) == 1:
                return {
                    "success": False, "failure_class": "worker_protocol_error",
                    "worker_returncode": -9, "worker_signal": "SIGKILL",
                    "error": "JSON worker vuoto",
                }
            return self.success(ctx)

        result, events = self.run_fake_search(fake_probe)
        self.assertEqual(calls, [64, 128, 129, 129, 129, 129])
        self.assertEqual(result["confirmed_max_gpu_resident_n_ctx"], 129)
        self.assertTrue(result["measurement_completed"])
        self.assertEqual(result["highest_gpu_resident_n_ctx_observed"], 129)
        self.assertEqual(len([
            e for e in events if e.get("event") == "context_worker_retry"
        ]), 1)

    def test_resource_failure_is_a_boundary_without_protocol_retry(self):
        calls = []

        def fake_probe(path, ctx, **kwargs):
            calls.append(ctx)
            if ctx == 129:
                return {
                    "success": False,
                    "failure_class": "native_cuda_resource_allocation_failure",
                    "worker_returncode": -6,
                    "worker_signal": "SIGABRT",
                    "error": "CUDA resource allocation failed at cuBLAS",
                }
            return self.success(ctx)

        result, events = self.run_fake_search(fake_probe)
        self.assertEqual(calls, [64, 128, 129, 128, 128])
        self.assertEqual(result["cuda_resource_failure_boundaries"], [129])
        self.assertEqual(result["confirmed_max_gpu_resident_n_ctx"], 128)
        self.assertEqual(result["highest_gpu_resident_n_ctx_observed"], 128)
        self.assertEqual(len(result["confirmation"]), 2)
        self.assertTrue(result["success"])
        self.assertFalse(any(
            e["event"] == "context_worker_retry" for e in events
        ))

    def test_two_protocol_failures_stop_without_inventing_maximum(self):
        calls = []
        def fake_probe(path, ctx, **kwargs):
            calls.append(ctx)
            if ctx == 129:
                return {
                    "success": False, "failure_class": "worker_protocol_error",
                    "worker_returncode": -9, "worker_signal": "SIGKILL",
                    "error": "JSON worker vuoto",
                }
            return self.success(ctx)

        result, events = self.run_fake_search(fake_probe)
        self.assertEqual(calls, [64, 128, 129, 129])
        self.assertIsNone(result["confirmed_max_gpu_resident_n_ctx"])
        self.assertEqual(result["highest_gpu_resident_n_ctx_observed"], 128)
        self.assertFalse(result["measurement_completed"])
        self.assertIn("worker_protocol_error", result["error"])
        self.assertEqual(len([
            e for e in events if e.get("event") == "context_worker_retry"
        ]), 1)


if __name__ == "__main__":
    unittest.main()
