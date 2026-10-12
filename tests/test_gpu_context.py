"""CPU-only tests for diagnostic maximal GPU-resident n_ctx discovery."""

import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from calibration.gpu_context import (
    context_limit, make_context_tokens, probe_context, search_gpu_context,
    verify_gpu_runtime,
)


def native_log():
    return {
        "native_buffer_records": [
            {"native_buffer": "CUDA0", "category": "model",
             "domain": "gpu_vram", "reported_mib": 5000.0},
            {"native_buffer": "CUDA0", "category": "kv",
             "domain": "gpu_vram", "reported_mib": 100.0},
            {"native_buffer": "CUDA0", "category": "compute",
             "domain": "gpu_vram", "reported_mib": 120.0},
            {"native_buffer": "CPU", "category": "output",
             "domain": "host_cpu", "reported_mib": 0.2},
        ],
    }


class GPUContextValidationTests(unittest.TestCase):
    def test_kv_and_compute_require_native_cuda_logs(self):
        result = verify_gpu_runtime(native_log())
        self.assertTrue(result["verified"])
        self.assertEqual(result["kv"]["gpu_buffer_records"], 1)
        for kind in ("kv", "compute"):
            with self.subTest(kind=kind):
                data = native_log()
                data["native_buffer_records"] = [
                    r for r in data["native_buffer_records"]
                    if r["category"] != kind
                ]
                self.assertFalse(verify_gpu_runtime(data)["verified"])

    def test_host_kv_fails_but_auxiliary_cuda_host_compute_is_reported(self):
        for kind in ("kv", "compute"):
            with self.subTest(kind=kind):
                data = native_log()
                data["native_buffer_records"].append({
                    "native_buffer": "CUDA_Host",
                    "category": kind,
                    "domain": "host_cuda_pinned",
                    "reported_mib": 8.0,
                })
                result = verify_gpu_runtime(data)
                self.assertEqual(result["verified"], kind == "compute")
                self.assertEqual(result[kind]["other_buffer_records"], 1)
                self.assertEqual(result[kind]["non_gpu_buffers"][0]["reported_mib"], 8.0)

    def test_real_context_prompt_occupies_requested_tokens(self):
        class FakeLLM:
            def tokenize(self, data, add_bos, special):
                self_check = list(data)
                return [1] + self_check

        llama = FakeLLM()
        for arch in ("gemma2", "llama", "qwen2"):
            with self.subTest(arch=arch):
                tokens = make_context_tokens(llama, arch, 512, 8)
                self.assertEqual(len(tokens), 503)
                self.assertEqual(tokens[0], 1)

    def test_probe_context_needs_real_gpu_logs_and_full_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "mock.gguf"
            model.write_bytes(b"mock")
            log = (
                "llama_kv_cache_init: CUDA0 KV buffer size = 12.0 MiB\n"
                "llama_new_context: CUDA0 compute buffer size = 40.0 MiB\n"
                "llama_new_context: CUDA_Host compute buffer size = 3.0 MiB\n"
            )
            worker = {
                "success": True, "n_ctx": 128,
                "measurements": {
                    "generated_tokens": 8, "prompt_tokens": 119,
                    "vram_peak_sampled_mib": 6000.0, "nvml_errors": [],
                },
            }
            completed = subprocess.CompletedProcess(
                ["python"], 0, stdout=json.dumps(worker), stderr=log
            )
            placement = {
                "verified": True, "tensor_count": 4, "unverified_names": [],
                "native_override": {"native_logged_occurrence_count": 4},
            }
            with (
                patch("calibration.gpu_context.inspect_gguf",
                      return_value={"metadata": {"general.architecture": "llama"}}),
                patch("calibration.gpu_context.native_confirmation",
                      return_value=placement),
                patch("calibration.gpu_context.subprocess.run",
                      return_value=completed),
            ):
                result = probe_context(model, 128)
            self.assertTrue(result["success"])
            self.assertTrue(result["gpu_runtime_validation"]["verified"])
            self.assertEqual(
                result["gpu_runtime_validation"]["compute"]["other_buffer_records"], 1
            )

            completed.stderr = log.replace(
                "CUDA0 KV buffer size", "CUDA_Host KV buffer size"
            )
            with (
                patch("calibration.gpu_context.inspect_gguf",
                      return_value={"metadata": {"general.architecture": "llama"}}),
                patch("calibration.gpu_context.native_confirmation",
                      return_value=placement),
                patch("calibration.gpu_context.subprocess.run",
                      return_value=completed),
            ):
                failed = probe_context(model, 128)
            self.assertFalse(failed["success"])
            self.assertEqual(failed["failure_class"],
                             "runtime_not_proven_gpu_resident")

    def test_incomplete_generation_never_establishes_gpu_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "mock.gguf"
            model.write_bytes(b"mock")
            log = (
                "CUDA0 KV buffer size = 8.0 MiB\n"
                "CUDA0 compute buffer size = 10.0 MiB\n"
            )
            worker = {
                "success": True, "n_ctx": 64,
                "measurements": {
                    "generated_tokens": 7, "prompt_tokens": 55,
                    "vram_peak_sampled_mib": 6000.0, "nvml_errors": [],
                },
            }
            with (
                patch("calibration.gpu_context.inspect_gguf",
                      return_value={"metadata": {}}),
                patch("calibration.gpu_context.native_confirmation",
                      return_value={
                          "verified": True, "tensor_count": 2,
                          "unverified_names": [],
                          "native_override": {"native_logged_occurrence_count": 2},
                      }),
                patch("calibration.gpu_context.subprocess.run",
                      return_value=subprocess.CompletedProcess(
                          ["python"], 0, stdout=json.dumps(worker), stderr=log
                      )),
            ):
                result = probe_context(model, 64)
            self.assertFalse(result["success"])
            self.assertEqual(result["failure_class"], "inference_or_worker_error")

    def test_context_limit_uses_architecture_metadata(self):
        for arch in ("gemma2", "llama", "qwen2"):
            self.assertEqual(context_limit({
                "metadata": {
                    "general.architecture": arch,
                    f"{arch}.context_length": 8192,
                },
            }), 8192)
        with self.assertRaises(ValueError):
            context_limit({"metadata": {"general.architecture": "llama"}})

    def test_full_gpu_model_manifest_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("calibration.gpu_context.GPUManager.calibration_status",
                      return_value={"ready": True, "memory": {"total_mib": 6141}}),
            ):
                with self.assertRaises(RuntimeError):
                    search_gpu_context("llama-3.1-8b-instruct", root=tmp)

    def test_doubling_then_bisection_confirms_highest_feasible(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models" / "llama-3.1-8b-instruct"
            model_dir.mkdir(parents=True)
            model = model_dir / "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
            model.write_bytes(b"mock")
            (model_dir / "baseline_selection.json").write_text(json.dumps({
                "model_id": "llama-3.1-8b-instruct",
                "architecture": "llama",
                "filename": model.name,
                "quantization": "Q5_K_M",
                "native_verified_all_gpu": True,
            }))
            calls = []
            def fake_probe(path, ctx, **kwargs):
                calls.append(ctx)
                if ctx <= 150:
                    return {
                        "success": True, "failure_class": None,
                        "measurements": {"vram_peak_sampled_mib": 6000.0},
                    }
                return {
                    "success": False, "failure_class": "native_cuda_oom",
                    "error": "cudaMalloc failed: out of memory",
                }

            with (
                patch("calibration.gpu_context.GPUManager.calibration_status",
                      return_value={"ready": True,
                                    "memory": {"total_mib": 6141.0}}),
                patch("calibration.gpu_context.inspect_gguf",
                      return_value={"metadata": {
                          "general.architecture": "llama",
                          "llama.context_length": 160,
                      }}),
                patch("calibration.gpu_context.probe_context",
                      side_effect=fake_probe),
            ):
                result = search_gpu_context(
                    "llama-3.1-8b-instruct",
                    root=Path(tmp) / "models", min_ctx=64,
                    confirm_rounds=2,
                )
            self.assertTrue(result["success"])
            self.assertEqual(result["confirmed_max_gpu_resident_n_ctx"], 150)
            self.assertEqual(result["highest_context_tested_and_failed"], 151)
            self.assertEqual(result["confirmation_min_sampled_headroom_mib"], 141.0)
            self.assertEqual(calls[:3], [64, 128, 160])
            self.assertEqual(calls[-2:], [150, 150])
            self.assertFalse(result["profile_generated"])
            self.assertIsNone(result["selected_operational_n_ctx"])
            marker = json.loads((model_dir / "gpu_context_reference.json").read_text())
            self.assertEqual(marker["n_ctx"], 150)
            self.assertFalse(marker["operationally_safe"])
            self.assertNotIn("max_gpu_resident_n_ctx",
                             json.loads((model_dir / "baseline_selection.json").read_text()))

    def test_unexplained_failure_does_not_claim_a_context_maximum(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "gemma-2-9b-it"
            model_dir.mkdir()
            model = model_dir / "gemma-2-9b-it-Q4_K_M.gguf"
            model.write_bytes(b"mock")
            (model_dir / "baseline_selection.json").write_text(json.dumps({
                "architecture": "gemma2", "filename": model.name,
                "quantization": "Q4_K_M", "native_verified_all_gpu": True,
            }))
            def fake_probe(path, ctx, **kwargs):
                if ctx == 64:
                    return {"success": True, "failure_class": None}
                return {"success": False,
                        "failure_class": "runtime_not_proven_gpu_resident"}
            with (
                patch("calibration.gpu_context.GPUManager.calibration_status",
                      return_value={"ready": True,
                                    "memory": {"total_mib": 6141.0}}),
                patch("calibration.gpu_context.inspect_gguf",
                      return_value={"metadata": {
                          "general.architecture": "gemma2",
                          "gemma2.context_length": 8192,
                      }}),
                patch("calibration.gpu_context.probe_context",
                      side_effect=fake_probe),
            ):
                result = search_gpu_context("gemma-2-9b-it", root=tmp)
            self.assertFalse(result["success"])
            self.assertIsNone(result["confirmed_max_gpu_resident_n_ctx"])
            self.assertIn("inconclusiva", result["error"])
            self.assertFalse((model_dir / "gpu_context_reference.json").exists())

    def test_ambiguous_oom_requires_explicit_optin_for_binary_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "gemma-2-9b-it"
            model_dir.mkdir()
            model = model_dir / "gemma-2-9b-it-Q4_K_M.gguf"
            model.write_bytes(b"mock")
            (model_dir / "baseline_selection.json").write_text(json.dumps({
                "architecture": "gemma2", "filename": model.name,
                "quantization": "Q4_K_M", "native_verified_all_gpu": True,
            }))
            def fake_probe(path, ctx, **kwargs):
                if ctx <= 100:
                    return {"success": True, "failure_class": None,
                            "measurements": {"vram_peak_sampled_mib": 6100.0}}
                return {
                    "success": False,
                    "failure_class": "ambiguous_context_initialization",
                    "error": "Failed to create llama_context",
                }

            common = [
                patch("calibration.gpu_context.GPUManager.calibration_status",
                      return_value={"ready": True,
                                    "memory": {"total_mib": 6141.0}}),
                patch("calibration.gpu_context.inspect_gguf",
                      return_value={"metadata": {
                          "general.architecture": "gemma2",
                          "gemma2.context_length": 256,
                      }}),
                patch("calibration.gpu_context.probe_context",
                      side_effect=fake_probe),
            ]
            with common[0], common[1], common[2]:
                strict = search_gpu_context("gemma-2-9b-it", root=tmp)
                self.assertFalse(strict["success"])
                allowed = search_gpu_context(
                    "gemma-2-9b-it", root=tmp,
                    retry_ambiguous_context=True,
                )
            self.assertTrue(allowed["success"])
            self.assertEqual(allowed["confirmed_max_gpu_resident_n_ctx"], 100)
            self.assertTrue(allowed["ambiguous_boundary_used"])


if __name__ == "__main__":
    unittest.main()
