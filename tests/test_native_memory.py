"""CPU-only tests for interpreting llama.cpp model/compute/KV memory logs."""

import unittest

from calibration.native_memory import parse_native_memory
from calibration.output_memory_compare import summarize_memory_audit


class NativeMemoryAuditTests(unittest.TestCase):
    def test_parse_model_and_runtime_gpu_memory(self):
        logs = (
            "llama_model_load_tensors: CUDA0 model buffer size = 4800.25 MiB\n"
            "llama_model_load_tensors: CUDA_Host model buffer size = 700.00 MiB\n"
            "llama_model_load_tensors: CPU_Mapped model buffer size = 200.00 MiB\n"
            "llama_kv_cache_init: CUDA0 KV buffer size = 22.50 MiB\n"
            "llama_context: CUDA0 compute buffer size = 15.00 MiB\n"
            "llama_context: CUDA0 output buffer size = 3.50 MiB\n"
        )
        r = parse_native_memory(logs)
        self.assertEqual(r["record_count"], 6)
        self.assertEqual(r["gpu_model_buffer_mib"], 4800.25)
        self.assertEqual(r["host_model_buffer_mib"], 900)
        self.assertEqual(
            r["totals_mib_by_category_and_domain"]["kv"]["gpu_vram"], 22.50
        )
        self.assertEqual(
            r["totals_mib_by_category_and_domain"]["compute"]["gpu_vram"], 15.0
        )

    def test_parser_not_guessing_missing_logs(self):
        r = parse_native_memory("CUDA device initialized\n")
        self.assertEqual(r["record_count"], 0)
        self.assertEqual(r["gpu_model_buffer_mib"], 0)
        self.assertEqual(r["totals_mib_by_category_and_domain"], {})

    def test_summary_requires_trials_and_buffers(self):
        baseline = {
            "candidate_id": "baseline", "success": True,
            "decode_tps": 16.6, "prefill_tps": 174, "nvml_loaded_mib": 4933,
            "nvml_peak_mib": 4967, "native_gpu_model_mib": 4400,
            "native_host_model_mib": 200, "native_gpu_compute_mib": 128.5,
            "native_buffer_record_count": 5,
            "load_seconds": 1.2
        }
        host = dict(
            baseline, candidate_id="tied:cpu:token_embd.weight",
            decode_tps=13.2, nvml_loaded_mib=4933, nvml_peak_mib=4967,
            native_gpu_model_mib=3682, native_host_model_mib=918,
            native_gpu_compute_mib=846.5,
            load_seconds=13
        )
        summary = summarize_memory_audit([baseline, host], 1)
        self.assertEqual(summary["native_gpu_model_buffer_change_mib"], -718)
        self.assertEqual(summary["nvml_peak_change_mib"], 0)
        self.assertEqual(summary["native_gpu_compute_buffer_change_mib"], 718)
        self.assertEqual(summary["native_gpu_model_plus_compute_change_mib"], 0)
        self.assertTrue(summary["model_weight_savings_compensated_in_compute_buffer"])
        self.assertFalse(summary["nvml_peak_savings_observed"])
        with self.assertRaises(ValueError):
            summarize_memory_audit([baseline, dict(host, native_buffer_record_count=0)], 1)
        with self.assertRaises(ValueError):
            summarize_memory_audit([baseline], 1)

    def test_compensation_not_claimed_for_partial_change(self):
        baseline = {
            "candidate_id": "baseline", "success": True,
            "decode_tps": 20, "prefill_tps": 200,
            "load_seconds": 1, "nvml_loaded_mib": 4900,
            "nvml_peak_mib": 5000, "native_gpu_model_mib": 4000,
            "native_gpu_compute_mib": 100, "native_host_model_mib": 200,
            "native_buffer_record_count": 5,
        }
        host = dict(
            baseline, candidate_id="tied:cpu:token_embd.weight",
            nvml_peak_mib=4900, native_gpu_model_mib=3900,
            native_gpu_compute_mib=140,
        )
        report = summarize_memory_audit([baseline, host], 1)
        self.assertEqual(report["native_gpu_model_plus_compute_change_mib"], -60)
        self.assertFalse(report["model_weight_savings_compensated_in_compute_buffer"])
        self.assertTrue(report["nvml_peak_savings_observed"])



if __name__ == "__main__":
    unittest.main()
