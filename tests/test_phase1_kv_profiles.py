"""CPU-only regression coverage for Phase 1 KV precision/Flash experiments."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.gpu_context import (
    validate_kv_config, verify_native_kv_config, probe_context,
)
from calibration.kv_profiles_phase1 import (
    PROFILES, run_profiles, precision_dominates, STATUS_SKIP,
)


EXPECTED = (
    (False, "f16", "f16"),
    (False, "q8_0", "f16"),
    (False, "q4_0", "f16"),
    (True, "f16", "f16"),
    (True, "f16", "q8_0"),
    (True, "q8_0", "f16"),
    (True, "q8_0", "q8_0"),
    (True, "f16", "q4_0"),
    (True, "q4_0", "f16"),
    (True, "q8_0", "q4_0"),
    (True, "q4_0", "q8_0"),
    (True, "q4_0", "q4_0"),
)


class KvProfileTests(unittest.TestCase):
    def test_complete_twelve_profile_matrix_and_compatibility(self):
        self.assertEqual(tuple(row[1:] for row in PROFILES), EXPECTED)
        self.assertEqual(len(set(row[0] for row in PROFILES)), 12)
        self.assertEqual(sum(not x[1] for x in PROFILES), 3)
        self.assertEqual(sum(x[1] for x in PROFILES), 9)
        for fa, k, v in EXPECTED:
            validate_kv_config(fa, k, v)
        with self.assertRaises(ValueError):
            validate_kv_config(False, "q8_0", "q4_0")
        with self.assertRaises(ValueError):
            validate_kv_config(True, "q3_k", "f16")

    def test_native_log_must_match_all_three_parameters(self):
        stderr = (
            "llama_context: flash_attn = 1\n"
            "llama_kv_cache: size = 700 MiB, K (q8_0): 450 MiB, "
            "V (q4_0): 250 MiB\n"
        )
        proved = verify_native_kv_config(
            stderr, flash_attn=True, type_k="q8_0", type_v="q4_0"
        )
        self.assertTrue(proved["verified"])
        self.assertFalse(verify_native_kv_config(
            stderr, flash_attn=True, type_k="q4_0", type_v="q8_0",
        )["verified"])
        self.assertFalse(verify_native_kv_config(
            stderr.replace("flash_attn = 1", "flash_attn = 0"),
            flash_attn=True, type_k="q8_0", type_v="q4_0",
        )["verified"])
        self.assertFalse(verify_native_kv_config(
            "llama_kv_cache: CUDA0 KV buffer size = 8.00 MiB",
            flash_attn=True, type_k="q8_0", type_v="q4_0",
        )["verified"])

    def test_worker_command_contains_explicit_kv_flags(self):
        import subprocess
        native = {"native_buffer_records": []}
        placement = {
            "verified": False, "tensor_count": 0,
            "unverified_names": [],
            "native_override": {"native_logged_occurrence_count": 0},
        }
        with (patch("calibration.gpu_context.inspect_gguf",
                    return_value={"tensors": [], "metadata": {
                        "general.architecture": "gemma2",
                        "gemma2.context_length": 8192,
                    }}),
              patch("calibration.gpu_context.subprocess.run",
                    return_value=subprocess.CompletedProcess(
                        [], 1, "", "unsupported KV profile"
                    )) as run,
              patch("calibration.gpu_context.parse_native_memory",
                    return_value=native),
              patch("calibration.gpu_context.native_confirmation",
                    return_value=placement)):
            probe_context(
                Path("/fake.gguf"), 512, generated_tokens=16,
                flash_attn=True, type_k="q8_0", type_v="q4_0",
            )
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--flash-attn") + 1], "on")
        self.assertEqual(cmd[cmd.index("--type-k") + 1], "q8_0")
        self.assertEqual(cmd[cmd.index("--type-v") + 1], "q4_0")
        self.assertEqual(cmd[cmd.index("--probe-n-ctx") + 1], "512")

    def test_sweep_passes_profile_and_never_overwrites_reference(self):
        calls = []
        def fake_search(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return {
                "success": True, "confirmed_max_gpu_resident_n_ctx": 900,
                "confirmation": [{"success": True}, {"success": True}],
            }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "kv.json"
            report = run_profiles(output=output, search=fake_search)
            self.assertTrue(report["all_profiles_completed"])
            self.assertEqual(len(calls), 11)
            for (_, kw), (_, flash, k, v) in zip(calls, PROFILES[1:]):
                self.assertEqual(
                    (kw["flash_attn"], kw["type_k"], kw["type_v"]),
                    (flash, k, v),
                )
                self.assertFalse(kw["persist_reference"])
                self.assertEqual(kw["tokens"], 16)
                self.assertEqual(kw["min_ctx"], 258)
            self.assertTrue(output.exists())
            self.assertEqual(report["profiles"][0]["status"], "reference_reused")
            self.assertEqual(report["profiles"][0]["max_gpu_n_ctx"], 257)
            self.assertTrue(all(r["max_gpu_n_ctx"] == 900
                                for r in report["profiles"][1:]))
            self.assertEqual(len(run_profiles(
                output=output, search=fake_search
            )["profiles"]), 12)
            self.assertEqual(len(calls), 11)  # completed profiles reused

    def test_one_failed_profile_does_not_hide_later_success(self):
        calls = []
        def fake_search(model_id, **kwargs):
            calls.append(kwargs["type_k"])
            if len(calls) == 2:
                raise RuntimeError("native kernel unsupported")
            return {"success": True, "confirmed_max_gpu_resident_n_ctx": 512}
        with tempfile.TemporaryDirectory() as directory:
            report = run_profiles(
                output=Path(directory) / "kv.json", search=fake_search
            )
            self.assertEqual(len(report["profiles"]), 12)
            self.assertIsNone(report["profiles"][2]["max_gpu_n_ctx"])
            self.assertEqual(report["profiles"][2]["status"],
                             "unsupported_or_inconclusive")
            self.assertEqual(report["profiles"][3]["max_gpu_n_ctx"], 512)


    def test_baseline_ctx_changes_start_and_report_identity(self):
        observed = []
        def fake_search(model_id, **kwargs):
            observed.append(kwargs["min_ctx"])
            return {"success": False, "attempts": [{
                "requested_n_ctx": kwargs["min_ctx"],
                "success": False,
                "failure_class": "native_cuda_oom",
            }]}
        with tempfile.TemporaryDirectory() as directory:
            result = run_profiles(
                baseline_ctx=300, output=Path(directory) / "kv.json",
                search=fake_search,
            )
        self.assertEqual(observed, [301] * 11)
        self.assertEqual(result["parameters"]["baseline_ctx"], 300)
        self.assertEqual(result["parameters"]["min_ctx"], 301)
        self.assertEqual(result["original_reference"]["n_ctx"], 300)
        self.assertEqual(len(result["profiles"]), 12)
        self.assertEqual(result["profiles"][0]["max_gpu_n_ctx"], 300)
        self.assertTrue(all(p["max_gpu_n_ctx"] is None
                            for p in result["profiles"][1:]))

    def test_first_failed_probe_does_not_search_higher_context(self):
        from calibration.gpu_context import search_gpu_context
        from model import ModelManager
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mock.gguf"
            path.write_bytes(b"mock")
            model_id = "gemma-2-9b-it"
            source = "bartowski"
            spec = ModelManager.source_spec(model_id, source)
            baseline = {
                "native_verified_all_gpu": True,
                "model_id": model_id,
                "source": source,
                "repository": spec["repository"],
                "filename": path.name,
                "local_path": str(path),
                "architecture": "gemma2",
                "quantization": "Q4_K_M",
            }
            status = {"ready": True, "errors": [],
                      "memory": {"total_mib": 6141}}
            calls = []
            def failed_probe(model_path, n_ctx, **kwargs):
                calls.append(n_ctx)
                return {
                    "success": False,
                    "failure_class": "native_cuda_oom",
                    "error": "CUDA out of memory",
                }
            with (patch("calibration.gpu_context.GPUManager.calibration_status",
                        return_value=status),
                  patch("calibration.gpu_context.inspect_gguf",
                        return_value={"metadata": {
                            "general.architecture": "gemma2",
                            "gemma2.context_length": 8192,
                        }}),
                  patch("calibration.gpu_context.probe_context",
                        side_effect=failed_probe)):
                result = search_gpu_context(
                    model_id, source=source, min_ctx=258, tokens=16,
                    baseline_selection=baseline, persist_reference=False,
                )
        self.assertFalse(result["success"])
        self.assertEqual(calls, [258])
        self.assertEqual(result["attempts"][0]["requested_n_ctx"], 258)


    def test_dominance_is_componentwise_and_same_flash_only(self):
        superior = {
            "flash_attn": True, "type_k": "q8_0", "type_v": "q8_0",
        }
        self.assertTrue(precision_dominates(
            superior, ("other", True, "q8_0", "q4_0")
        ))
        self.assertTrue(precision_dominates(
            superior, ("other", True, "q4_0", "q8_0")
        ))
        self.assertTrue(precision_dominates(
            superior, ("other", True, "q4_0", "q4_0")
        ))
        self.assertFalse(precision_dominates(
            superior, ("other", True, "f16", "q4_0")
        ))
        self.assertFalse(precision_dominates(
            superior, ("other", True, "q4_0", "f16")
        ))
        self.assertFalse(precision_dominates(
            superior, ("other", False, "q4_0", "q4_0")
        ))

    def test_q8_q8_model_ceiling_prunes_only_inferior_quality(self):
        calls = []
        def fake_search(model_id, **kwargs):
            k, v, fa = kwargs["type_k"], kwargs["type_v"], kwargs["flash_attn"]
            calls.append((fa, k, v))
            maximum = 8192 if (fa and k == v == "q8_0") else 7000
            return {
                "success": True, "confirmed_max_gpu_resident_n_ctx": maximum,
                "model_declared_context_length": 8192,
            }
        with tempfile.TemporaryDirectory() as directory:
            report = run_profiles(
                output=Path(directory) / "kv.json", search=fake_search,
            )
        by_id = {p["id"]: p for p in report["profiles"]}
        for k, v in (
            ("q8_0", "q4_0"), ("q4_0", "q8_0"), ("q4_0", "q4_0")
        ):
            row = by_id[f"fa_on_k_{k}_v_{v}"]
            self.assertEqual(row["status"], STATUS_SKIP)
            self.assertEqual(row["dominated_by"], "fa_on_k_q8_0_v_q8_0")
            self.assertNotIn((True, k, v), calls)
        self.assertIn((True, "f16", "q4_0"), calls)
        self.assertIn((True, "q4_0", "f16"), calls)
        self.assertEqual(len(calls), 8)
        self.assertTrue(report["all_profiles_completed"])

    def test_skip_requires_model_limit_not_arbitrary_max_ctx(self):
        calls = []
        def fake_search(model_id, **kwargs):
            calls.append((kwargs["flash_attn"],
                          kwargs["type_k"], kwargs["type_v"]))
            return {
                "success": True, "confirmed_max_gpu_resident_n_ctx": 1024,
                "model_declared_context_length": 8192,
            }
        with tempfile.TemporaryDirectory() as directory:
            report = run_profiles(
                max_ctx=1024,
                output=Path(directory) / "kv.json",
                search=fake_search,
            )
        self.assertEqual(len(calls), 11)
        self.assertFalse(any(p["status"] == STATUS_SKIP
                             for p in report["profiles"]))

    def test_direct_probe_rejects_beyond_gguf_without_cuda_worker(self):
        with (patch("calibration.gpu_context.inspect_gguf",
                    return_value={"metadata": {
                        "general.architecture": "gemma2",
                        "gemma2.context_length": 8192,
                    }}),
              patch("calibration.gpu_context.subprocess.run") as run):
            with self.assertRaisesRegex(ValueError, "8192"):
                probe_context(
                    Path("/fake.gguf"), 8193, generated_tokens=16,
                )
        run.assert_not_called()

    def test_model_limit_is_never_exceeded_during_bounded_search(self):
        from calibration.gpu_context import search_gpu_context
        from model import ModelManager
        model_id = "gemma-2-9b-it"
        source = "bartowski"
        spec = ModelManager.source_spec(model_id, source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mock.gguf"
            path.write_bytes(b"mock")
            selection = {
                "native_verified_all_gpu": True,
                "model_id": model_id, "source": source,
                "repository": spec["repository"],
                "filename": path.name, "local_path": str(path),
                "architecture": "gemma2", "quantization": "Q4_K_M",
            }
            calls = []
            def succeeds(model_path, ctx, **kwargs):
                calls.append(ctx)
                return {
                    "success": True, "n_ctx": ctx,
                    "measurements": {"vram_peak_sampled_mib": 5500},
                }
            with (patch("calibration.gpu_context.GPUManager.calibration_status",
                        return_value={"ready": True, "errors": [],
                                      "memory": {"total_mib": 6141}}),
                  patch("calibration.gpu_context.inspect_gguf",
                        return_value={"metadata": {
                            "general.architecture": "gemma2",
                            "gemma2.context_length": 8192,
                        }}),
                  patch("calibration.gpu_context.probe_context",
                        side_effect=succeeds)):
                result = search_gpu_context(
                    model_id, source=source, baseline_selection=selection,
                    min_ctx=258, max_ctx=100000, persist_reference=False,
                )
        self.assertTrue(result["success"])
        self.assertEqual(result["confirmed_max_gpu_resident_n_ctx"], 8192)
        self.assertEqual(result["search_ceiling"], 8192)
        self.assertTrue(all(258 <= ctx <= 8192 for ctx in calls))
        self.assertIn(8192, calls)



if __name__ == "__main__":
    unittest.main()
