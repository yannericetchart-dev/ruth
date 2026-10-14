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


def fixture(root, *, accepted=True):
    original = select_model("production")
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
                 pid=3456, mem=5800, exitcode=0, error=None):
        self.calls = []
        self.layers, self.total = layers, total
        self.observed, self.tokens = observed, tokens
        self.pid, self.mem, self.exitcode = pid, mem, exitcode
        self.error = error

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        assert cmd[1:3] == ["-I", "-c"]
        assert cmd[3] == COMMAND
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == UUID
        assert kw["env"]["PYTHONNOUSERSITE"] == "1"
        assert kw["timeout"] == 360
        data = {
            "schema": SCHEMA, "purpose": "production",
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
            cmd, self.exitcode, PREFIX + json.dumps(data) + "\n", log,
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
    def run_fake(self, root, *, worker=None, accepted=True):
        choice, path, env = fixture(root, accepted=accepted)
        process = worker if worker is not None else FakeProcess()
        with patch("ruth_runtime.production_probe.select_model",
                   return_value=choice):
            result = probe_production_candidate(
                project_root=root, authorize_gpu_inference=True,
                environment=env, run=process,
                verify_native=lambda **k: SimpleNamespace(
                    status="native_library_dual_abi_verified"),
                discover_gpu=lambda **k: (UUID, PCI),
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

    def test_rejects_test_purpose_for_production_probe(self):
        with patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main([
                "models", "probe-production", "--purpose", "test"
            ]), 2)
            self.assertIn("catalog_selection_blocked", output.getvalue())


if __name__ == "__main__":
    unittest.main()
