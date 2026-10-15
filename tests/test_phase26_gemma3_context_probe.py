"""Gemma3 context staged-probe regression tests; no CUDA or real GGUF loaded."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from ruth_runtime.context_source_evidence import (
    calculate_context_source, context_source_derivation,
)
from ruth_runtime.gemma3_context_gate import validate_gemma3_context_request
from ruth_runtime.load_only_probe import probe_model_load
from ruth_runtime.load_only_worker import run_load_only, native_kv_dtype
from ruth_runtime.install_plan import WHEEL_VERSION

GPU_UUID = "GPU-test-pinned-context"

BASE = {
    "n_ctx": 256, "n_batch": 64, "n_ubatch": 64,
    "swa_full": True, "flash_attention": "off",
    "type_k": "F16", "type_v": "F16", "cuda_preinit": False,
}


class Gemma3ContextGateTests(unittest.TestCase):
    def test_only_two_native_allocations_are_allowlisted(self):
        for flash, dtype in (("off", "F16"), ("on", "BF16")):
            result = validate_gemma3_context_request(
                **{**BASE, "flash_attention": flash,
                   "type_k": dtype, "type_v": dtype})
            self.assertTrue(result["experimental_allocation_only"])
            self.assertFalse(result["production_approved"])
            self.assertFalse(result["inference_approved"])
        for bad in (
            {"n_ctx": 512}, {"n_ctx": 257}, {"n_batch": 32},
            {"n_ubatch": 32}, {"swa_full": False},
            {"cuda_preinit": True},
            {"flash_attention": "auto"},
            {"flash_attention": "on", "type_k": "F16", "type_v": "F16"},
            {"flash_attention": "off", "type_k": "BF16", "type_v": "BF16"},
            {"flash_attention": "on", "type_k": "BF16", "type_v": "F16"},
            {"flash_attention": "on", "type_k": "F32", "type_v": "F32"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_gemma3_context_request(**{**BASE, **bad})

    def test_source_cache_geometry_and_evidence_remain_logical(self):
        for flash, dtype in (("off", "F16"), ("on", "BF16")):
            with self.subTest(dtype=dtype):
                kwargs = {**BASE, "flash_attention": flash,
                          "type_k": dtype, "type_v": dtype}
                kwargs.pop("cuda_preinit")
                source = calculate_context_source(
                    model_version="gemma-3-1b-it", **kwargs)
                derived = context_source_derivation(source)
                self.assertEqual(derived["global_cells_derived"], 256)
                self.assertEqual(derived["swa_cells_derived"], 256)
                self.assertEqual(derived["key_logical_bytes_derived"], 3407872)
                self.assertEqual(derived["value_logical_bytes_derived"], 3407872)
                self.assertIsNone(derived["swa_full_off_saves_bytes_derived"])
                self.assertIsNone(derived["swa_off_first_saving_at_effective_ctx"])
                self.assertTrue(derived["model_geometry_verification_pending"])
                self.assertFalse(source["native_allocation_executed"])

    def test_parent_rejects_unauthorized_or_unqualified_before_environment(self):
        result = probe_model_load(
            model_version="gemma-3-1b-it", quantization="BF16",
            mode="context", n_ctx=256, allow_gpu_load=False,
        )
        self.assertEqual(result["status"], "gpu_load_not_authorized")
        self.assertFalse(result["inference_executed"])
        with patch(
            "ruth_runtime.load_only_probe.inspect_python_environment",
            side_effect=AssertionError("No environment or CUDA may be inspected"),
        ):
            for bad in ({"n_ctx": 512}, {"flash_attention": "auto"},
                        {"n_batch": 32}, {"cuda_preinit": True},
                        {"flash_attention": "off", "type_k": "BF16", "type_v": "BF16"},
                        {"flash_attention": "on", "type_k": "F16", "type_v": "F16"}):
                with self.subTest(bad=bad):
                    probe = probe_model_load(
                        model_version="gemma-3-1b-it", quantization="BF16",
                        mode="context", allow_gpu_load=True,
                        **{**BASE, **bad})
                    self.assertEqual(probe["status"], "unsupported_kv_context_configuration")

    def test_enum_bf16_is_pinned_to_native_ggml_header(self):
        self.assertEqual(native_kv_dtype(SimpleNamespace(), "BF16"), 30)
        self.assertEqual(native_kv_dtype(
            SimpleNamespace(GGML_TYPE_BF16=30), "BF16"), 30)
        with self.assertRaisesRegex(RuntimeError, "enum mismatch"):
            native_kv_dtype(SimpleNamespace(GGML_TYPE_BF16=29), "BF16")


class Gemma3WorkerTests(unittest.TestCase):
    def worker(self, flash: str, dtype: str):
        class Api:
            GGML_TYPE_F16 = 1
            LLAMA_FLASH_ATTN_TYPE_DISABLED = 0
            LLAMA_FLASH_ATTN_TYPE_ENABLED = 1
            def __init__(self):
                self.calls = []
                self.context_options = None
            def llama_supports_gpu_offload(self):
                return True
            def llama_backend_init(self):
                self.calls.append("backend_init")
            def llama_backend_free(self):
                self.calls.append("backend_free")
            def llama_model_default_params(self):
                return SimpleNamespace(n_gpu_layers=0, main_gpu=-1, use_mmap=False)
            def llama_model_load_from_file(self, path, params):
                self.calls.append(("model_load", params.n_gpu_layers))
                return object()
            def llama_model_free(self, model):
                self.calls.append("model_free")
            def llama_context_default_params(self):
                return SimpleNamespace()
            def llama_init_from_model(self, model, params):
                self.calls.append("context_init")
                self.context_options = params
                return object()
            def llama_free(self, context):
                self.calls.append("context_free")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = root / "pinned.gguf"
            f.write_bytes(b"GGUFdata")
            choice = SimpleNamespace(
                family="gemma", model="gemma-3",
                version="gemma-3-1b-it", quantization="BF16",
                expected_size_bytes=f.stat().st_size,
            )
            api = Api()
            package = ModuleType("llama_cpp")
            package.__version__ = WHEEL_VERSION
            package.llama_cpp = api
            options = {
                "n_batch": 64, "n_ubatch": 64, "swa_full": True,
                "flash_attention": flash, "type_k": dtype, "type_v": dtype,
            }
            with patch.dict(sys.modules, {"llama_cpp": package}), \
                 patch("ruth_runtime.load_only_worker.select_model",
                       return_value=choice), \
                 patch("ruth_runtime.load_only_worker._local_path",
                       return_value=f), \
                 patch("ruth_runtime.load_only_worker.measure_worker_gpu"), \
                 patch("ruth_runtime.load_only_worker.current_cuda_free_mib",
                       return_value=4200.0), \
                 patch.dict("os.environ", {
                     "CUDA_VISIBLE_DEVICES": GPU_UUID,
                     "RUTH_LLAMA_KV_OPTIONS": json.dumps(options),
                 }):
                result = run_load_only(
                    root, GPU_UUID, "BF16", "context", n_ctx=256,
                    model_version="gemma-3-1b-it")
            self.assertTrue(result["weights_loaded"])
            self.assertTrue(result["context_initialized"])
            self.assertFalse(result["inference_executed"])
            self.assertEqual(result["generated_tokens"], 0)
            self.assertEqual(api.calls.count("context_init"), 1)
            self.assertIn(("model_load", 2**31 - 1), api.calls)
            self.assertEqual(api.context_options.type_k, 30 if dtype == "BF16" else 1)
            self.assertEqual(api.context_options.type_v, 30 if dtype == "BF16" else 1)
            self.assertEqual(api.context_options.flash_attn_type,
                             1 if flash == "on" else 0)
            self.assertFalse(api.context_options.kv_unified)
            self.assertEqual(api.context_options.n_seq_max, 1)
            return result

    def test_native_worker_f16_flash_off_never_generates(self):
        self.worker("off", "F16")

    def test_native_worker_bf16_flash_on_never_generates(self):
        self.worker("on", "BF16")


if __name__ == "__main__":
    unittest.main()
