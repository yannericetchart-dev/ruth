"""Offline regression: native abort evidence survives singleton aggregation.

In particular SIGABRT before the child writes JSON must NOT lose its
stderr/cublas reason, which was the stage-2 checkpoint regression.
"""
import unittest

from calibration.kv_adaptive_tensor_offload import (
    _probe_summary, summarize_probe_repeats,
)
from calibration.worker_failure import analyze_worker_failure


class AbortCheckpointTests(unittest.TestCase):
    def test_sigabrt_inference_preserves_native_stderr_in_checkpoint_sample(self):
        stderr = (
            "RUTH_WORKER_PHASE=load_begin\n"
            "RUTH_WORKER_PHASE=load_complete\n"
            "RUTH_WORKER_PHASE=prefill_begin\n"
            "CUDA error: the resource allocation failed\n"
            "  current device: 0, in function cublas_handle\n"
        )
        diagnosis = analyze_worker_failure(stderr, None, -6, False)
        native_probe = {
            "success": False,
            "failure_class": "native_cuda_resource_allocation_failure",
            "error": "worker stdout JSON non valido",
            "worker_returncode": -6,
            "worker_signal": "SIGABRT",
            "worker_json_valid": False,
            "measurements": {},
            **diagnosis,
        }
        sample = _probe_summary(native_probe)
        aggregate = summarize_probe_repeats([sample], 2)
        self.assertFalse(aggregate["success"])
        self.assertEqual(aggregate["repetitions"], 1)
        self.assertEqual(
            aggregate["failure_class"],
            "native_cuda_resource_allocation_failure",
        )
        saved = aggregate["samples"][0]
        self.assertEqual(saved["worker_signal"], "SIGABRT")
        self.assertEqual(saved["worker_returncode"], -6)
        self.assertIs(saved["worker_json_valid"], False)
        details = saved["failure_diagnostics"]
        self.assertEqual(details["failed_during"], "prefill")
        self.assertTrue(details["abort_signal_detected"])
        self.assertTrue(details["native_cuda_resource_allocation_failed"])
        self.assertFalse(details["suspected_cuda_oom"])
        self.assertIn("cublas_handle", details["stderr_tail"])
        self.assertIn(
            "CUDA error: the resource allocation failed",
            details["fatal_native_lines"],
        )

    def test_absent_native_data_is_not_fabricated(self):
        sample = _probe_summary({
            "success": False, "failure_class": "timeout",
            "error": "worker did not return",
        })
        self.assertEqual(sample["failure_diagnostics"], {})
        self.assertIsNone(sample["worker_json_valid"])
        self.assertIsNone(sample["worker_signal"])


if __name__ == "__main__":
    unittest.main()
