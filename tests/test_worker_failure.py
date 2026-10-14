"""Offline proof that native CUDA error evidence is not conflated with OOM."""

import unittest
from unittest.mock import patch
import json
import subprocess

from calibration.benchmark import benchmark
from calibration.worker_failure import analyze_worker_failure


class WorkerFailureTests(unittest.TestCase):
    def test_graceful_context_failure_is_ambiguous_not_oom(self):
        report = analyze_worker_failure(
            "llama_new_context_with_model: failed to create llama_context",
            "ValueError: Failed to create llama_context",
            0, False,
        )
        self.assertEqual(report["failure_hint"], "ambiguous_context_initialization")
        self.assertTrue(report["context_creation_failed"])
        self.assertFalse(report["suspected_cuda_oom"])
        self.assertIn("llama_context", report["stderr_tail"])

    def test_explicit_native_cuda_allocation_error_is_oom(self):
        report = analyze_worker_failure(
            "ggml_cuda: cudaMalloc failed: out of memory",
            "ValueError: Failed to create llama_context",
            -6, False,
        )
        self.assertEqual(report["failure_hint"], "explicit_cuda_oom_evidence")
        self.assertTrue(report["suspected_cuda_oom"])
        self.assertEqual(len(report["cuda_diagnostics"]), 1)

    def test_cublas_handle_allocation_failure_is_explicit_resource_boundary(self):
        stderr = (
            "CUDA error: the resource allocation failed\n"
            "  current device: 0, in function cublas_handle\n"
            "  cublasCreate_v2(&cublas_handles[device][curr_stream_no])\n"
            "CUDA error"
        )
        outcome = analyze_worker_failure(
            stderr, "JSON worker vuoto", -6, False
        )
        self.assertEqual(
            outcome["failure_hint"],
            "explicit_cuda_resource_allocation_failure",
        )
        self.assertTrue(outcome["native_cuda_resource_allocation_failed"])
        self.assertFalse(outcome["suspected_cuda_oom"])
        self.assertIn("cublas_handle", outcome["stderr_tail"])

    def test_unrelated_sigabrt_and_generic_resource_string_not_allocation(self):
        for text in ("SIGABRT: process aborted", "resource allocation warning",
                     "CUDA error: invalid device function"):
            with self.subTest(stderr=text):
                outcome = analyze_worker_failure(
                    text, "JSON worker vuoto", -6, False
                )
                self.assertIsNone(outcome["failure_hint"])
                self.assertFalse(outcome["native_cuda_resource_allocation_failed"])
                self.assertFalse(outcome["suspected_cuda_oom"])

    def test_cublas_status_alloc_failed_is_not_mislabelled_oom(self):
        outcome = analyze_worker_failure(
            "cuBLAS error: CUBLAS_STATUS_ALLOC_FAILED",
            "JSON worker vuoto", -6, False,
        )
        self.assertTrue(outcome["native_cuda_resource_allocation_failed"])
        self.assertFalse(outcome["suspected_cuda_oom"])

    def test_sigabrt_alone_is_not_oom(self):
        report = analyze_worker_failure(
            "worker aborted",
            "Worker JSON invalid",
            -6, False,
        )
        self.assertFalse(report["suspected_cuda_oom"])
        self.assertIsNone(report["failure_hint"])

    def test_clean_success_with_native_warnings_is_not_oom(self):
        report = analyze_worker_failure(
            "older unrelated warning: out of memory", None, 0, True,
        )
        self.assertFalse(report["suspected_cuda_oom"])
        self.assertIsNone(report["stderr_tail"])

    def test_stderr_is_bounded_without_losing_tail(self):
        report = analyze_worker_failure(
            "logline\n" * 1000 + "llama_context failed to initialize",
            "ValueError: Failed to create llama_context", 0, False,
        )
        self.assertLessEqual(len(report["stderr_tail"]), 6000)
        self.assertIn("failed to initialize", report["stderr_tail"])
        self.assertLessEqual(len(report["cuda_diagnostics"]), 24)

    def test_abort_captures_native_phase_and_diagnostics(self):
        raw = (
            "RUTH_WORKER_PHASE=load_begin\n"
            "RUTH_WORKER_PHASE=load_complete\n"
            "RUTH_WORKER_PHASE=prefill_begin\n"
            "CUDA error: the resource allocation failed\n"
            "  current device: 0, in function cublas_handle\n"
        )
        outcome = analyze_worker_failure(raw, None, -6, False)
        self.assertTrue(outcome["abort_signal_detected"])
        self.assertEqual(outcome["last_worker_phase"], "prefill_begin")
        self.assertEqual(outcome["failed_during"], "prefill")
        self.assertTrue(outcome["native_cuda_resource_allocation_failed"])
        self.assertFalse(outcome["suspected_cuda_oom"])
        self.assertIn("resource allocation failed", "\n".join(
            outcome["fatal_native_lines"]
        ))
        self.assertIn("RUTH_WORKER_PHASE=prefill_begin",
                      outcome["stderr_tail"])

    def test_successful_probe_is_not_treated_as_native_abort(self):
        report = analyze_worker_failure(
            "RUTH_WORKER_PHASE=decode_complete\n", None, 0, True,
        )
        self.assertIsNone(report["failed_during"])
        self.assertFalse(report["abort_signal_detected"])
        self.assertIsNone(report["stderr_tail"])

    def test_benchmark_retains_native_logs_for_exit_zero_failure(self):
        completed = subprocess.CompletedProcess(
            ["python"], 0,
            stdout=json.dumps({
                "success": False,
                "error": "ValueError: Failed to create llama_context",
            }),
            stderr="llama_context: failed to allocate compute buffer",
        )
        with patch("calibration.benchmark.subprocess.run", return_value=completed):
            result = benchmark(
                gpu_layers=39, max_tokens=256,
                probe_n_ctx=2048, prompt_repeat=34,
            )
        self.assertEqual(result["worker_returncode"], 0)
        self.assertEqual(result["failure_hint"], "ambiguous_context_initialization")
        self.assertFalse(result["suspected_cuda_oom"])
        self.assertIn("failed to allocate", result["stderr_tail"])
        self.assertTrue(result["native_evidence_available"])


if __name__ == "__main__":
    unittest.main()
