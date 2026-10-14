"""Offline tests: pinned Gemma production download/128-token CUDA probe.

Uses tiny fake GGUF and subprocesses; never downloads weights or starts CUDA.
"""
from dataclasses import replace
import hashlib
import json
from io import StringIO
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from ruth_runtime.model_catalog import select_model, load_catalog
from ruth_runtime.native_failure_diagnostics import diagnose_cuda_failure
from ruth_runtime.model_download import _local_path, _manifest, download_pinned_model
from ruth_runtime.production_probe import (
    COMMAND, N_CTX, PREFIX, probe_production_candidate,
)
from ruth_runtime.production_probe_worker import (
    SCHEMA, run_production_worker,
)
from ruth_runtime.launcher import main

UUID = "GPU-5a841828-882b-85d9-c98e-3a0d29e66e00"
PCI = "00000000:01:00.0"
GGUF = b"GGUF" + b"gemma synthetic mock" * 12


def fixture(root, *, accepted=True, quantization="Q4_K_L"):
    original = select_model("production", quantization=quantization)
    choice = replace(
        original, expected_size_bytes=len(GGUF),
        expected_sha256=hashlib.sha256(GGUF).hexdigest(),
    )
    file = _local_path(root, choice)
    file.parent.mkdir(parents=True)
    file.write_bytes(GGUF)
    sidecar = {
        **_manifest(choice), "verified": True,
        "license_acknowledged_for_this_download": accepted,
    }
    file.with_name(file.name + ".manifest.json").write_text(json.dumps(sidecar))
    env = SimpleNamespace(
        interpreter_in_project_venv=True, python_version="3.12.3",
        project_root=str(root), interpreter=str(root / ".venv/bin/python"),
    )
    return choice, file, env


class FakeProcess:
    def __init__(self, *, layers=51, total=51, observed=True, tokens=12,
                 pid=3456, mem=5800, exitcode=0, error=None,
                 native_log=None, emit_result=True, quantization="Q4_K_L"):
        self.calls = []
        self.layers, self.total = layers, total
        self.observed, self.tokens = observed, tokens
        self.pid, self.mem, self.exitcode = pid, mem, exitcode
        self.error = error
        self.native_log = native_log
        self.emit_result = emit_result
        self.quantization = quantization

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        assert cmd[1:3] == ["-I", "-c"]
        assert cmd[3] == COMMAND
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == UUID
        assert kw["env"]["PYTHONNOUSERSITE"] == "1"
        assert kw["timeout"] == 360
        assert cmd[-1] in ("Q4_K_L", "Q4_K_M")
        data = {
            "schema": SCHEMA, "purpose": "production",
            "quantization": self.quantization,
            "gpu_uuid": UUID, "worker_pid": self.pid,
            "model_loaded": True, "completion_tokens": self.tokens,
            "gpu_process_observed": self.observed,
            "gpu_memory_peak_mib_observed": self.mem,
            "n_ctx": N_CTX, "n_batch": 64, "n_ubatch": 64,
            "max_tokens": 24, "seed": 42,
            "worker_duration_seconds": 1.2,
            "cpu_fallback_selected": False,
            "model_cache_enabled": False,
        }
        if self.error:
            data["error"] = self.error
        log = f"llama_model_load_tensors: offloaded {self.layers}/{self.total} layers to GPU"
        return subprocess.CompletedProcess(
            cmd, self.exitcode,
            PREFIX + json.dumps(data) + "\n" if self.emit_result else "",
            log if self.native_log is None else self.native_log,
        )


class GemmaProductionPinTests(unittest.TestCase):
    def test_default_is_exact_pinned_q4_kl(self):
        choice = select_model("production")
        self.assertEqual(choice.quantization, "Q4_K_L")
        self.assertEqual(choice.revision,
                         "d731033f3dc4018261fd39896e50984d398b4ac5")
        self.assertEqual(choice.expected_size_bytes, 5983265728)
        self.assertEqual(choice.expected_sha256,
                         "ed8573f0d1e3ca58e471498fd162a65d4f4253dffdec28a7ba394fdcae92f1f3")
        self.assertFalse(choice.local_model_ready)
        self.assertFalse(choice.downloadable)

    def test_q4_km_fallback_kept_as_distinct_pin(self):
        gemma = next(e for e in load_catalog()["models"] if e["family"] == "gemma")
        self.assertEqual({v["quantization"] for v in gemma["variants"]},
                         {"Q4_K_L", "Q4_K_M"})
        other = next(v for v in gemma["variants"]
                     if v["quantization"] == "Q4_K_M")
        self.assertEqual(other["expected_size_bytes"], 5761057728)
        self.assertEqual(other["expected_sha256"],
                         "13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe")

    def test_download_requires_separate_consent_even_if_pinned(self):
        choice = select_model("production")
        report = download_pinned_model(choice, allow_download=False,
                                       acknowledge_model_license=False)
        self.assertEqual(report.status, "model_download_not_authorized")
        self.assertFalse(report.download_started)
        self.assertFalse(report.model_verified)


class GemmaMinimumContextProbeTests(unittest.TestCase):
    def run_fake(self, root, *, worker=None, accepted=True,
                 quantization="Q4_K_L"):
        choice, path, env = fixture(
            root, accepted=accepted, quantization=quantization,
        )
        process = worker if worker is not None else FakeProcess(
            quantization=quantization,
        )
        with patch("ruth_runtime.production_probe.select_model",
                   return_value=choice):
            result = probe_production_candidate(
                project_root=root, authorize_gpu_inference=True,
                quantization=quantization, environment=env, run=process,
                verify_native=lambda **k: SimpleNamespace(
                    status="native_library_dual_abi_verified"),
                discover_gpu=lambda **k: (UUID, PCI),
                gpu_preflight=lambda uuid, **kw: SimpleNamespace(
                    status="gpu_ready",
                    to_dict=lambda: {"status": "gpu_ready", "gpu_uuid": uuid,
                                     "memory_mib": {"total": 6141, "reserved": 369,
                                                    "used": 0, "free": 5772},
                                     "processes": [], "recommendations": []},
                ),
            )
        return result, process

    def test_no_gpu_authorization_blocks_before_file_or_hardware(self):
        with TemporaryDirectory() as temp:
            report = probe_production_candidate(
                project_root=Path(temp),
                authorize_gpu_inference=False,
                run=lambda *a, **k: self.fail("Must not run"),
            )
            self.assertEqual(report.status, "gpu_inference_not_authorized")
            self.assertFalse(report.download_started)

    def test_no_local_model_refuses_before_cuda(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            choice = select_model("production")
            env = SimpleNamespace(
                interpreter_in_project_venv=True, python_version="3.12.3",
                project_root=str(root), interpreter=str(root / ".venv/bin/python"),
            )
            with patch("ruth_runtime.production_probe.select_model",
                       return_value=choice):
                result = probe_production_candidate(
                    project_root=root, authorize_gpu_inference=True,
                    environment=env, run=lambda *a, **k: self.fail("No subprocess"),
                    verify_native=lambda **k: self.fail("No native work"),
                )
            self.assertEqual(result.status, "model_integrity_or_license_unverified")
            self.assertFalse(result.gpu_inference_verified)

    def test_manifest_without_license_ack_blocks(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            choice, file, env = fixture(root, accepted=False)
            with patch("ruth_runtime.production_probe.select_model",
                       return_value=choice):
                result = probe_production_candidate(
                    project_root=root, authorize_gpu_inference=True,
                    environment=env,
                    verify_native=lambda **k: self.fail("Do not verify native"),
                )
            self.assertEqual(result.status, "model_integrity_or_license_unverified")

    def test_tiny_verified_model_and_full_gpu_evidence_mock(self):
        with TemporaryDirectory() as temp:
            result, proc = self.run_fake(Path(temp))
            self.assertEqual(result.status, "gemma_min_context_gpu_verified")
            self.assertTrue(result.gpu_inference_verified)
            self.assertTrue(result.full_gpu_offload_verified)
            self.assertEqual(result.offloaded_layers, 51)
            self.assertEqual(result.generated_tokens, 12)
            self.assertEqual(result.n_ctx, 128)
            self.assertTrue(result.context_calibration_pending)
            self.assertFalse(result.runtime_profile_published)
            self.assertFalse(result.cpu_fallback_selected)
            self.assertEqual(len(proc.calls), 1)

    def test_xorg_blocks_before_cuda_and_reports_host_recommendations(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            choice, _, env = fixture(root)
            def blocked(uuid, **kwargs):
                return SimpleNamespace(
                    status="gpu_busy", error=None,
                    to_dict=lambda: {
                        "status": "gpu_busy",
                        "gpu_uuid": uuid,
                        "memory_mib": {
                            "total": 6141, "reserved": 369,
                            "used": 15, "free": 5759,
                        },
                        "processes": [{
                            "pid": 3032, "name": "/usr/lib/xorg/Xorg",
                            "type": "G", "gpu_memory_mib": 4,
                        }],
                        "recommendations": [
                            "Usare TTY o SSH, salvare il lavoro e valutare stop display-manager",
                        ],
                    },
                )
            with patch("ruth_runtime.production_probe.select_model",
                       return_value=choice):
                result = probe_production_candidate(
                    project_root=root, authorize_gpu_inference=True,
                    environment=env,
                    run=lambda *a, **kw: self.fail("No CUDA worker allowed"),
                    verify_native=lambda **k: SimpleNamespace(
                        status="native_library_dual_abi_verified"),
                    discover_gpu=lambda **k: (UUID, PCI),
                    gpu_preflight=blocked,
                )
            self.assertEqual(result.status, "gpu_busy")
            self.assertFalse(result.gpu_inference_verified)
            self.assertFalse(result.runtime_profile_published)
            self.assertEqual(result.gpu_preflight["processes"][0]["pid"], 3032)
            self.assertIn("TTY", result.gpu_preflight["recommendations"][0])

    def test_partial_offload_does_not_pass_even_with_tokens(self):
        with TemporaryDirectory() as temp:
            result, _ = self.run_fake(Path(temp),
                                       worker=FakeProcess(layers=49, total=51))
            self.assertEqual(result.status, "gemma_min_context_inconclusive")
            self.assertFalse(result.gpu_inference_verified)
            self.assertFalse(result.full_gpu_offload_verified)

    def test_missing_gpu_pid_observation_does_not_pass(self):
        with TemporaryDirectory() as temp:
            result, _ = self.run_fake(Path(temp),
                                       worker=FakeProcess(observed=False))
            self.assertEqual(result.status, "gemma_min_context_inconclusive")
            self.assertFalse(result.gpu_inference_verified)

    def test_worker_failure_does_not_claim_production(self):
        with TemporaryDirectory() as temp:
            result, _ = self.run_fake(
                Path(temp), worker=FakeProcess(exitcode=2,
                                               error="CUDA out of memory"))
            self.assertEqual(result.status, "production_min_context_worker_failed")
            self.assertIn("out of memory", result.error)
            self.assertFalse(result.runtime_profile_published)

    def test_native_cuda_oom_is_reported_from_real_native_signal(self):
        log = "\n".join((
            "llama_model_load_tensors: CUDA0 buffer size = 5812.50 MiB",
            "ggml_cuda: cudaMalloc failed: out of memory",
            "llama_model_load: failed to load model from /home/test/gguf/model.gguf",
        )) + "\n"
        with TemporaryDirectory() as temp:
            result, _ = self.run_fake(
                Path(temp), worker=FakeProcess(
                    exitcode=2, error="ValueError: Failed to load model from file",
                    native_log=log,
                )
            )
        self.assertEqual(result.status, "production_min_context_worker_failed")
        self.assertEqual(result.native_failure_category, "cuda_out_of_memory_reported")
        self.assertIn("explicit_out_of_memory", result.native_failure_signals)
        self.assertEqual(result.native_cuda_buffer_mib_reported, 5812.5)
        self.assertNotIn("/home/test", repr(result.native_log_excerpt))
        self.assertFalse(result.gpu_inference_verified)
        self.assertEqual(result.worker_exit_code, 2)

    def test_generic_python_load_error_does_not_prove_oom(self):
        with TemporaryDirectory() as temp:
            result, _ = self.run_fake(
                Path(temp), worker=FakeProcess(
                    exitcode=2,
                    error="ValueError: Failed to load model from file",
                    native_log="llama_model_load: failed to load model",
                ),
            )
        self.assertEqual(result.native_failure_category,
                         "native_model_load_failed_cause_unknown")
        self.assertNotIn("explicit_out_of_memory", result.native_failure_signals)
        self.assertFalse(result.gpu_inference_verified)

    def test_native_unsupported_gguf_is_not_misclassified_as_oom(self):
        with TemporaryDirectory() as temp:
            result, _ = self.run_fake(
                Path(temp), worker=FakeProcess(
                    exitcode=2, error="Failed to load model",
                    native_log="llama_model_loader: unsupported tensor type",
                ),
            )
        self.assertEqual(result.native_failure_category,
                         "gguf_or_backend_unsupported_reported")
        self.assertFalse(result.gpu_inference_verified)

    def test_crash_without_json_still_preserves_native_diagnostics(self):
        with TemporaryDirectory() as temp:
            result, _ = self.run_fake(
                Path(temp), worker=FakeProcess(
                    exitcode=-6, emit_result=False,
                    native_log="ggml_cuda: cudaMalloc failed: out of memory",
                ),
            )
        self.assertEqual(result.status, "worker_failed_or_no_result")
        self.assertEqual(result.native_failure_category, "cuda_out_of_memory_reported")
        self.assertEqual(result.worker_exit_code, -6)
        self.assertFalse(result.runtime_profile_published)

    def test_no_diagnostic_signal_has_unknown_classification_not_oom(self):
        result = diagnose_cuda_failure(
            "", worker_error="ValueError: Failed to load model from file",
            returncode=2,
        )
        self.assertEqual(result["native_failure_category"],
                         "native_failure_cause_unknown")
        self.assertTrue(result["generic_python_load_error_only"])
        self.assertEqual(result["native_failure_signals"], [])
        self.assertIsNone(result["native_cuda_buffer_mib_reported"])

    def test_native_log_excerpt_is_bounded_and_redacted(self):
        logs = "\n".join(
            "ggml_cuda: error loading /home/example/private/model.gguf"
            for _ in range(40)
        )
        report = diagnose_cuda_failure(logs, returncode=2)
        self.assertLessEqual(len(report["native_log_excerpt"]), 8)
        self.assertNotIn("/home/example", repr(report["native_log_excerpt"]))

    def test_explicit_q4_km_can_be_probed_without_changing_default(self):
        with TemporaryDirectory() as temp:
            result, worker = self.run_fake(
                Path(temp), quantization="Q4_K_M",
            )
            self.assertEqual(result.candidate, "gemma-2-9B-it/gguf/Q4_K_M")
            self.assertEqual(result.status, "gemma_min_context_gpu_verified")
            self.assertTrue(result.gpu_inference_verified)
            self.assertEqual(worker.calls[0][0][-1], "Q4_K_M")
            self.assertEqual(select_model("production").quantization, "Q4_K_L")

    def test_q4_km_missing_local_file_cannot_use_q4_kl_pinned_gguf(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, path, env = fixture(root, quantization="Q4_K_L")
            self.assertTrue(path.is_file())
            with patch("ruth_runtime.production_probe.select_model",
                       wraps=select_model):
                result = probe_production_candidate(
                    project_root=root, authorize_gpu_inference=True,
                    quantization="Q4_K_M", environment=env,
                    verify_native=lambda **kw: self.fail("No CUDA"),
                )
            self.assertEqual(result.status, "model_integrity_or_license_unverified")
            self.assertEqual(result.candidate, "gemma-2-9B-it/gguf/Q4_K_M")
            self.assertFalse(result.model_loaded)

    def test_unlisted_quantization_rejected_before_any_native_or_network(self):
        result = probe_production_candidate(
            authorize_gpu_inference=True, quantization="IQ4_XS",
            run=lambda *a, **kw: self.fail("No subprocess"),
        )
        self.assertEqual(result.status, "quantization_not_approved_for_probe")
        self.assertFalse(result.gpu_inference_verified)

    def test_worker_result_wrong_quantization_must_never_pass(self):
        with TemporaryDirectory() as temp:
            result, _ = self.run_fake(
                Path(temp), quantization="Q4_K_M",
                worker=FakeProcess(quantization="Q4_K_L"),
            )
            self.assertEqual(result.status, "gemma_min_context_probe_failed")
            self.assertFalse(result.gpu_inference_verified)

    def test_native_real_q4_kl_cuda_oom_text_classified_from_user_output(self):
        log = "\n".join((
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5700.31 MiB on device 0: cudaMalloc failed: out of memory",
            "llama_model_load: error loading model: unable to allocate CUDA0 buffer",
            "llama_model_load_from_file_impl: failed to load model",
        ))
        diagnosis = diagnose_cuda_failure(log, returncode=2)
        self.assertEqual(diagnosis["native_failure_category"],
                         "cuda_out_of_memory_reported")
        self.assertIn("explicit_out_of_memory",
                      diagnosis["native_failure_signals"])
        self.assertEqual(diagnosis["native_cuda_buffer_mib_reported"], 5700.31)
        self.assertEqual(diagnosis["worker_exit_code"], 2)

    def test_worker_hash_change_refused(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            choice, file, env = fixture(root)
            file.write_bytes(b"GGUFbad")
            with patch("ruth_runtime.production_probe.select_model",
                       return_value=choice):
                result = probe_production_candidate(
                    project_root=root, authorize_gpu_inference=True,
                    environment=env, verify_native=lambda **k: self.fail("No CUDA"))
            self.assertEqual(result.status, "model_integrity_or_license_unverified")


class GemmaProbeWorkerTests(unittest.TestCase):
    def test_worker_uses_min_context_and_captures_tokens(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            choice, _, env = fixture(root)
            class Session:
                def __init__(self):
                    self.closed = False
                def generate_report(self, prompt, *, max_tokens):
                    self.prompt, self.max_tokens = prompt, max_tokens
                    return {"completion_tokens": 9, "text": "Ciao"}
                def close(self): self.closed = True
            sess = Session()
            with patch("ruth_runtime.production_probe_worker.select_model",
                       return_value=choice), \
                 patch("ruth_runtime.production_probe_worker.CudaLlamaAdapter") as adapter, \
                 patch("ruth_runtime.production_probe_worker.measure_worker_gpu",
                       side_effect=lambda pid, uuid, stop, samples: samples.append(930)), \
                 patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": UUID}):
                adapter.return_value.load.return_value = sess
                report = run_production_worker(root, UUID)
                self.assertEqual(
                    adapter.return_value.load.call_args.args[2],
                    {"n_ctx": 128, "n_batch": 64, "n_ubatch": 64})
                self.assertTrue(adapter.call_args.kwargs["native_diagnostics"])
            self.assertTrue(sess.closed)
            self.assertEqual(report["completion_tokens"], 9)
            self.assertTrue(report["gpu_process_observed"])
            self.assertFalse(report["cpu_fallback_selected"])

    def test_worker_explicit_q4_km_uses_distinct_pinned_file(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            choice, path, _ = fixture(root, quantization="Q4_K_M")
            class Session:
                def generate_report(self, prompt, *, max_tokens):
                    return {"completion_tokens": 7, "text": "ciao"}
                def close(self):
                    self.closed = True
            sess = Session()
            with patch("ruth_runtime.production_probe_worker.select_model",
                       return_value=choice) as selector, \
                 patch("ruth_runtime.production_probe_worker.CudaLlamaAdapter") as adapter, \
                 patch("ruth_runtime.production_probe_worker.measure_worker_gpu",
                       side_effect=lambda pid, uuid, stop, samples: samples.append(1024)), \
                 patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": UUID}):
                adapter.return_value.load.return_value = sess
                result = run_production_worker(root, UUID, quantization="Q4_K_M")
                self.assertEqual(selector.call_args.kwargs["quantization"], "Q4_K_M")
                self.assertEqual(adapter.return_value.load.call_args.args[0], path)
            self.assertEqual(result["quantization"], "Q4_K_M")
            self.assertEqual(result["completion_tokens"], 7)
            self.assertTrue(result["gpu_process_observed"])
            self.assertTrue(sess.closed)

    def test_worker_rejects_unpinned_candidate_before_backend(self):
        with TemporaryDirectory() as temp:
            with patch("ruth_runtime.production_probe_worker.CudaLlamaAdapter",
                       side_effect=AssertionError("No model loading")), \
                 patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": UUID}):
                with self.assertRaises(RuntimeError):
                    run_production_worker(Path(temp), UUID, quantization="IQ4_XS")

    def test_worker_rejects_wrong_gpu_before_native_import(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            choice, _, _ = fixture(root)
            with patch("ruth_runtime.production_probe_worker.select_model",
                       return_value=choice), \
                 patch("ruth_runtime.production_probe_worker.CudaLlamaAdapter",
                       side_effect=AssertionError("No CUDA")), \
                 patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-wrong"}):
                with self.assertRaises(RuntimeError):
                    run_production_worker(root, UUID)


class ProductionProbeCliTests(unittest.TestCase):
    def test_no_flag_refuses(self):
        with patch("ruth_runtime.production_probe.probe_production_candidate") as probe, \
             patch("sys.stdout", new_callable=StringIO) as output:
            probe.return_value = SimpleNamespace(
                gpu_inference_verified=False,
                to_dict=lambda: {"status": "gpu_inference_not_authorized"})
            code = main(["models", "probe-production"])
            self.assertEqual(code, 2)
            self.assertFalse(probe.call_args.kwargs["authorize_gpu_inference"])
            self.assertIn("gpu_inference_not_authorized", output.getvalue())

    def test_flag_passed_only_on_explicit_command(self):
        with patch("ruth_runtime.production_probe.probe_production_candidate") as probe, \
             patch("sys.stdout", new_callable=StringIO):
            probe.return_value = SimpleNamespace(
                gpu_inference_verified=True,
                to_dict=lambda: {"status": "gemma_min_context_gpu_verified"})
            self.assertEqual(main([
                "models", "probe-production", "--allow-gpu-inference"
            ]), 0)
            self.assertTrue(probe.call_args.kwargs["authorize_gpu_inference"])
            self.assertEqual(probe.call_args.kwargs["quantization"], "Q4_K_L")

    def test_cli_explicit_q4_km_is_passed_to_production_probe(self):
        with (patch("ruth_runtime.production_probe.probe_production_candidate") as probe,
              patch("sys.stdout", new_callable=StringIO)):
            probe.return_value = SimpleNamespace(
                gpu_inference_verified=False,
                to_dict=lambda: {"status": "model_integrity_or_license_unverified"},
            )
            self.assertEqual(main([
                "models", "probe-production", "--purpose", "production",
                "--quantization", "Q4_K_M", "--allow-gpu-inference",
            ]), 2)
            self.assertEqual(probe.call_args.kwargs["quantization"], "Q4_K_M")
            self.assertTrue(probe.call_args.kwargs["authorize_gpu_inference"])

    def test_rejects_test_purpose_for_production_probe(self):
        with patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main([
                "models", "probe-production", "--purpose", "test"
            ]), 2)
            self.assertIn("catalog_selection_blocked", output.getvalue())


if __name__ == "__main__":
    unittest.main()
