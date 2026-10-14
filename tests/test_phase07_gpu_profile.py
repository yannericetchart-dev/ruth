"""Offline regression tests for repeatable CUDA GGUF performance profiles."""
from dataclasses import replace
import hashlib
import json
from io import StringIO
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruth_runtime.cuda_adapter import (
    CudaLlamaAdapter, INITIAL_STACK, UnsafeRuntimeOptions,
)
from ruth_runtime.gpu_profile import (
    PROFILE_COMMAND, PROFILE_PREFIX, PROFILES, _check_rows,
    run_gpu_profile, _empty,
)
from ruth_runtime.profiling_worker import (
    SAMPLES, WARMUP, CTX, TOKENS, SEED, _stats,
)
from ruth_runtime.launcher import main
from ruth_runtime.model_catalog import select_model
from ruth_runtime.model_download import _local_path, _manifest

DATA = b"GGUF" + b"baseline mock" * 16
UUID = "GPU-aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"
PCI = "00000000:01:00.0"


def fixture(root):
    choice = replace(select_model("test"),
                     expected_size_bytes=len(DATA),
                     expected_sha256=hashlib.sha256(DATA).hexdigest())
    file = _local_path(root, choice)
    file.parent.mkdir(parents=True)
    file.write_bytes(DATA)
    file.with_name(file.name + ".manifest.json").write_text(json.dumps({
        **_manifest(choice), "verified": True,
        "license_acknowledged_for_this_download": True,
    }))
    env = SimpleNamespace(
        interpreter_in_project_venv=True, python_version="3.12.3",
        project_root=str(root), interpreter=str(root / ".venv/bin/python"),
    )
    return choice, file, env


def round_metric(*, decode=24, prefill=0.1, decode_time=0.3):
    prompt = 48
    return {
        "prompt_tokens": prompt, "completion_tokens": decode + 1,
        "decode_tokens": decode, "eos": False,
        "tokenize_seconds": .001,
        "prefill_seconds": prefill,
        "first_token_seconds": .002,
        "decode_seconds": decode_time,
        "prefill_tokens_per_second": prompt / prefill,
        "decode_tokens_per_second": decode / decode_time,
    }


class FakeProfile:
    def __init__(self, *, layers=True, memory=True, tokens=True,
                 success=True, tamper=None):
        self.calls = []
        self.layers = layers
        self.memory = memory
        self.tokens = tokens
        self.success = success
        self.tamper = tamper

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        assert cmd[1:3] == ["-I", "-c"]
        assert cmd[3] == PROFILE_COMMAND
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == UUID
        assert kwargs["timeout"] == 480
        rows = [round_metric() for _ in range(SAMPLES)]
        if not self.tokens:
            rows[0]["completion_tokens"] = 0
        data = {
            "schema": "ruth-cuda-profile-v1",
            "purpose": "test", "gpu_uuid": UUID,
            "model_loaded": True, "worker_pid": 12345,
            "gpu_process_observed": self.memory,
            "gpu_memory_peak_mib_observed": 921 if self.memory else None,
            "samples_requested": SAMPLES, "samples_completed": SAMPLES,
            "warmup_rounds": WARMUP, "n_ctx": CTX,
            "max_tokens": TOKENS, "seed": SEED,
            "n_gpu_layers_requested": -1, "cache_enabled": False,
            "rounds": rows, "summary": _stats(rows),
            "model_load_seconds": .25,
        }
        if self.tamper:
            data.update(self.tamper)
        stdout = PROFILE_PREFIX + json.dumps(data) + "\n"
        stderr = "llama_model_load_tensors: offloaded 17/17 layers to GPU" \
            if self.layers else "CPU allocation only"
        return subprocess.CompletedProcess(
            cmd, 0 if self.success else 2, stdout, stderr,
        )


def trusted(**_):
    return SimpleNamespace(status="native_library_dual_abi_verified")


class ProfileParentTests(unittest.TestCase):
    def go(self, root, *, fake=None, allow=True, purpose="test",
           native=trusted, saving=False):
        model, _, env = fixture(root)
        worker = fake if fake is not None else FakeProfile()
        with patch("ruth_runtime.gpu_profile.select_model", return_value=model):
            result = run_gpu_profile(
                project_root=root, purpose=purpose,
                authorize_gpu_inference=allow, environment=env,
                run=worker, native_check=native,
                gpu_check=lambda **kw: (UUID, PCI), save=saving,
            )
        return result, worker

    def test_no_permission_no_filesystem_or_gpu_probe(self):
        result = run_gpu_profile(
            project_root=Path("/nonexistent"), authorize_gpu_inference=False,
            run=lambda *a, **kw: self.fail("GPU accessed"),
        )
        self.assertEqual(result.status, "gpu_profiling_not_authorized")
        self.assertFalse(result.gpu_profile_verified)

    def test_parent_recomputes_median_min_max(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            result, worker = self.go(root)
            self.assertEqual(result.status, "gpu_profile_verified")
            self.assertTrue(result.gpu_profile_verified)
            self.assertTrue(result.model_inference_executed)
            self.assertEqual(result.samples, 5)
            self.assertEqual(result.warmup_rounds, 1)
            self.assertEqual(result.offloaded_layers, 17)
            self.assertEqual(result.statistics["decode_tokens_per_second"]["median"], 80.)
            self.assertEqual(result.model_load_seconds, .25)
            self.assertEqual(result.gpu_memory_peak_mib_observed, 921)
            self.assertFalse(result.cpu_fallback_selected)
            self.assertEqual(len(worker.calls), 1)

    def test_report_saved_json_with_model_sha_and_revision(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            result, _ = self.go(root, saving=True)
            self.assertTrue(result.gpu_profile_verified)
            saved = Path(result.saved_report)
            self.assertTrue(saved.is_file())
            self.assertEqual(saved.parent, root / PROFILES)
            record = json.loads(saved.read_text())
            self.assertEqual(record["profile_schema"], "ruth-cuda-profile-result-v1")
            self.assertEqual(record["samples"], 5)
            self.assertEqual(record["model_revision"],
                             "067b946cf014b7c697f3654f621d577a3e3afd1c")
            self.assertTrue(len(record["model_sha256"]) == 64)

    def test_unverified_lib_blocks_worker(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            result, fake = self.go(root, native=lambda **_: SimpleNamespace(status="wrong"))
            self.assertEqual(result.status, "native_runtime_unverified")
            self.assertEqual(fake.calls, [])

    def test_cuda_layers_absent_is_inconclusive(self):
        with TemporaryDirectory() as temp:
            result, _ = self.go(Path(temp), fake=FakeProfile(layers=False))
            self.assertEqual(result.status, "gpu_profile_inconclusive")
            self.assertFalse(result.gpu_profile_verified)

    def test_missing_gpu_pid_memory_is_inconclusive(self):
        with TemporaryDirectory() as temp:
            result, _ = self.go(Path(temp), fake=FakeProfile(memory=False))
            self.assertEqual(result.status, "gpu_profile_inconclusive")

    def test_bad_sample_count_fail_closed(self):
        with TemporaryDirectory() as temp:
            result, _ = self.go(Path(temp),
                                fake=FakeProfile(tamper={"samples_completed": 4}))
            self.assertEqual(result.status, "gpu_profile_failed")

    def test_wrong_uuid_refused(self):
        with TemporaryDirectory() as temp:
            result, _ = self.go(Path(temp),
                                fake=FakeProfile(tamper={"gpu_uuid": "GPU-other"}))
            self.assertEqual(result.status, "gpu_profile_inconclusive")

    def test_fewer_than_eight_decode_tokens_refused(self):
        with TemporaryDirectory() as temp:
            result, _ = self.go(Path(temp), fake=FakeProfile(tokens=False))
            self.assertEqual(result.status, "gpu_profile_failed")

    def test_worker_nonzero_exit_never_claims_success(self):
        with TemporaryDirectory() as temp:
            result, _ = self.go(Path(temp), fake=FakeProfile(success=False))
            self.assertEqual(result.status, "gpu_profile_worker_failed")
            self.assertFalse(result.model_inference_executed)

    def test_production_default_not_qualified(self):
        result = run_gpu_profile(
            project_root=Path("/nonexistent"), purpose="production",
            authorize_gpu_inference=True,
        )
        self.assertEqual(result.status, "gpu_production_profiling_not_authorized")

    def test_invalid_venv_refused(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            result = run_gpu_profile(
                project_root=root, authorize_gpu_inference=True,
                environment=SimpleNamespace(
                    interpreter_in_project_venv=False, python_version="3.12.3",
                    project_root=str(root), interpreter="/usr/bin/python3",
                ),
            )
            self.assertEqual(result.status, "invalid_project_venv")

    def test_parent_detects_timing_rate_inconsistency(self):
        rows = [round_metric() for _ in range(SAMPLES)]
        rows[0]["decode_tokens_per_second"] = 999
        with self.assertRaises(ValueError):
            _check_rows({
                "schema": "ruth-cuda-profile-v1", "purpose": "test",
                "samples_requested": SAMPLES, "samples_completed": SAMPLES,
                "warmup_rounds": WARMUP, "n_ctx": CTX,
                "max_tokens": TOKENS, "seed": SEED,
                "n_gpu_layers_requested": -1, "cache_enabled": False,
                "rounds": rows,
            })

    def test_missing_profile_dir_symlink_is_fail_closed(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / "outside"
            outside.mkdir()
            (root / ".ruth-build").mkdir()
            (root / PROFILES).symlink_to(outside, target_is_directory=True)
            result, _ = self.go(root, saving=True)
            self.assertEqual(result.status, "gpu_profile_failed")
            self.assertEqual(list(outside.iterdir()), [])


class FakeModel:
    cache = None
    def __init__(self, **kwargs):
        self.args = kwargs
        self.eval_calls = []
        self.reset_calls = 0
        self.sample_calls = 0
    def n_ctx(self): return 512
    def tokenize(self, text, add_bos=True):
        return list(range(10))
    def reset(self): self.reset_calls += 1
    def set_seed(self, seed): self.seed = seed
    def eval(self, tokens): self.eval_calls.append(list(tokens))
    def sample(self, *, temp):
        assert temp == 0.0
        self.sample_calls += 1
        return 100 + self.sample_calls
    def token_eos(self): return -1
    def close(self): self.closed = True


class SessionBenchmarkTests(unittest.TestCase):
    def test_real_phase_boundaries_invoked_and_seed_fixed(self):
        with TemporaryDirectory() as temp:
            gguf = Path(temp) / "tiny.gguf"
            gguf.write_bytes(b"GGUFfake")
            fake = FakeModel()
            backend = SimpleNamespace(
                __version__="0.3.36",
                llama_supports_gpu_offload=lambda: True,
                Llama=lambda **kwargs: fake,
            )
            session = CudaLlamaAdapter(
                load_backend=lambda name: backend, seed=42,
            ).load(gguf, INITIAL_STACK, {})
            result = session.benchmark_round("fixed prompt", max_tokens=24)
            self.assertEqual(result["completion_tokens"], 24)
            self.assertEqual(result["decode_tokens"], 23)
            self.assertEqual(result["prompt_tokens"], 10)
            self.assertEqual(fake.eval_calls[0], list(range(10)))
            self.assertEqual(len(fake.eval_calls), 24)
            self.assertEqual(fake.seed, 42)
            self.assertEqual(fake.reset_calls, 1)
            self.assertGreater(result["prefill_seconds"], 0)
            self.assertGreater(result["decode_seconds"], 0)
            session.close()

    def test_cache_changes_block_benchmark(self):
        with TemporaryDirectory() as temp:
            gguf = Path(temp) / "tiny.gguf"
            gguf.write_bytes(b"fake")
            fake = FakeModel()
            backend = SimpleNamespace(
                __version__="0.3.36",
                llama_supports_gpu_offload=lambda: True,
                Llama=lambda **kwargs: fake,
            )
            session = CudaLlamaAdapter(load_backend=lambda _: backend).load(
                gguf, INITIAL_STACK, {})
            fake.cache = object()
            with self.assertRaises(UnsafeRuntimeOptions):
                session.benchmark_round("prompt", max_tokens=24)
            session.close()


class ProfileCliTests(unittest.TestCase):
    def test_explicit_flag_required(self):
        result = SimpleNamespace(
            gpu_profile_verified=False,
            to_dict=lambda: {"status": "gpu_profiling_not_authorized"},
        )
        with patch("ruth_runtime.gpu_profile.run_gpu_profile",
                   return_value=result) as profile, \
             patch("sys.stdout", new_callable=StringIO) as out:
            self.assertEqual(main(["models", "profile", "--purpose", "test"]), 2)
            self.assertFalse(profile.call_args.kwargs["authorize_gpu_inference"])
            self.assertIn("gpu_profiling_not_authorized", out.getvalue())

    def test_explicit_test_profile_flag_reaches_profiling(self):
        result = SimpleNamespace(
            gpu_profile_verified=True,
            to_dict=lambda: {"status": "gpu_profile_verified"},
        )
        with patch("ruth_runtime.gpu_profile.run_gpu_profile",
                   return_value=result) as profile, \
             patch("sys.stdout", new_callable=StringIO) as out:
            self.assertEqual(main(["models", "profile", "--purpose", "test",
                                   "--allow-gpu-inference"]), 0)
            self.assertTrue(profile.call_args.kwargs["authorize_gpu_inference"])
            self.assertIn("gpu_profile_verified", out.getvalue())


if __name__ == "__main__":
    unittest.main()
