"""No-disk-cache CUDA adapter contract tests, without installed llama_cpp."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from ruth_runtime.contracts import StackSelection
from ruth_runtime.cuda_adapter import (
    CudaLlamaAdapter, INITIAL_STACK,
    UnsafeRuntimeOptions, UnverifiedCudaEngine,
)
from ruth_runtime.install_plan import WHEEL_VERSION


class FakeLlama:
    instances = []
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cache = None
        self.closed = False
        self.inferences = []
        FakeLlama.instances.append(self)

    def create_completion(self, **kwargs):
        self.inferences.append(kwargs)
        return {"choices": [{"text": "cuda mock (NOT native evidence)"}]}

    def close(self):
        self.closed = True


def backend(*, offload=True, version=WHEEL_VERSION, constructor=FakeLlama):
    return SimpleNamespace(
        __version__=version,
        llama_supports_gpu_offload=lambda: offload,
        Llama=constructor,
    )


class CudaAdapterTests(unittest.TestCase):
    def setUp(self):
        FakeLlama.instances.clear()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name) / "test.gguf"
        self.model.write_bytes(b"NOT A REAL MODEL")
        self.import_calls = []

    def factory(self, b=None):
        engine = b if b is not None else backend()
        def importer(name):
            self.import_calls.append(name)
            self.assertEqual(name, "llama_cpp")
            return engine
        return CudaLlamaAdapter(load_backend=importer)

    def test_default_cache_none_and_gpu_offload_forced(self):
        session = self.factory().load(self.model, INITIAL_STACK, {})
        self.assertTrue(session.cache_disabled)
        self.assertEqual(self.import_calls, ["llama_cpp"])
        engine = FakeLlama.instances[-1]
        self.assertEqual(engine.kwargs["n_gpu_layers"], -1)
        self.assertEqual(engine.kwargs["main_gpu"], 0)
        self.assertFalse(engine.kwargs["vocab_only"])
        self.assertNotIn("cache", engine.kwargs)
        self.assertNotIn("cache_dir", engine.kwargs)
        self.assertEqual(session.generate("ciao", max_tokens=3),
                         "cuda mock (NOT native evidence)")
        self.assertEqual(engine.inferences, [
            {"prompt": "ciao", "max_tokens": 3, "stream": False}
        ])
        session.close()
        self.assertTrue(engine.closed)

    def test_cache_disk_and_cpu_injection_never_reach_backend(self):
        malicious_options = [
            {"cache": True}, {"cache": None},
            {"cache_dir": "/tmp/untrusted"},
            {"diskcache": object()}, {"LlamaDiskCache": "fake"},
            {"n_gpu_layers": 0}, {"n_gpu_layers": -1},
            {"main_gpu": 1}, {"backend": "cpu"}, {"use_mmap": False},
            {"disk": "path"}, {"pickle": True},
        ]
        adapter = self.factory()
        for options in malicious_options:
            with self.subTest(options=list(options)):
                with self.assertRaises(UnsafeRuntimeOptions):
                    adapter.load(self.model, INITIAL_STACK, options)
        self.assertEqual(self.import_calls, [])
        self.assertEqual(FakeLlama.instances, [])

    def test_wrong_stack_denied_before_import(self):
        fake = StackSelection("cpu", "intel", "cpu", "llama.cpp", "llama_cpp_python")
        with self.assertRaises(UnverifiedCudaEngine):
            self.factory().load(self.model, fake, {})
        self.assertEqual(self.import_calls, [])

    def test_binding_without_gpu_offload_denied(self):
        with self.assertRaises(UnverifiedCudaEngine):
            self.factory(backend(offload=False)).load(self.model, INITIAL_STACK, {})
        self.assertEqual(FakeLlama.instances, [])

    def test_wrong_binding_version_denied(self):
        with self.assertRaises(UnverifiedCudaEngine):
            self.factory(backend(version="0.3.35")).load(self.model, INITIAL_STACK, {})
        self.assertEqual(FakeLlama.instances, [])

    def test_unexpected_native_cache_closes_engine_and_fails(self):
        class UnsafeLlama(FakeLlama):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.cache = object()
        with self.assertRaises(UnsafeRuntimeOptions):
            self.factory(backend(constructor=UnsafeLlama)).load(
                self.model, INITIAL_STACK, {})
        self.assertTrue(FakeLlama.instances[-1].closed)

    def test_cache_changed_after_creation_blocks_generation(self):
        session = self.factory().load(self.model, INITIAL_STACK, {})
        FakeLlama.instances[-1].cache = object()
        with self.assertRaises(UnsafeRuntimeOptions):
            session.generate("ciao")
        self.assertEqual(FakeLlama.instances[-1].inferences, [])
        session.close()

    def test_bad_model_path_blocks_native_backend_import(self):
        bad = Path(self.tmp.name) / "missing.gguf"
        with self.assertRaises(ValueError):
            self.factory().load(bad, INITIAL_STACK, {})
        symlink = Path(self.tmp.name) / "alias.gguf"
        symlink.symlink_to(self.model)
        with self.assertRaises(ValueError):
            self.factory().load(symlink, INITIAL_STACK, {})
        self.assertEqual(self.import_calls, [])

    def test_gpu_options_require_consistent_bounds(self):
        cases = [
            {"n_ctx": 0}, {"n_batch": True}, {"n_ubatch": -1},
            {"n_ctx": 100, "n_batch": 101},
            {"n_batch": 64, "n_ubatch": 65},
        ]
        for opts in cases:
            with self.subTest(opts=opts):
                with self.assertRaises(UnsafeRuntimeOptions):
                    self.factory().load(self.model, INITIAL_STACK, opts)
        self.assertEqual(self.import_calls, [])

    def test_valid_explicit_context_does_not_enable_cache(self):
        session = self.factory().load(
            self.model, INITIAL_STACK,
            {"n_ctx": 1024, "n_batch": 256, "n_ubatch": 128})
        engine = FakeLlama.instances[-1]
        self.assertEqual(engine.kwargs["n_ctx"], 1024)
        self.assertEqual(engine.kwargs["n_batch"], 256)
        self.assertEqual(engine.kwargs["n_ubatch"], 128)
        self.assertTrue(session.cache_disabled)
        session.close()

    def test_explicit_flash_swa_and_independent_kv_types_reach_python_binding(self):
        session = self.factory().load(
            self.model, INITIAL_STACK,
            {"n_ctx": 256, "n_batch": 64, "n_ubatch": 32,
             "flash_attn": True, "swa_full": False,
             "type_k": "Q8_0", "type_v": "Q4_0"})
        args = FakeLlama.instances[-1].kwargs
        self.assertTrue(args["flash_attn"])
        self.assertFalse(args["swa_full"])
        self.assertEqual(args["type_k"], 8)
        self.assertEqual(args["type_v"], 2)
        self.assertEqual(args["n_batch"], 64)
        self.assertEqual(args["n_ubatch"], 32)
        self.assertNotIn("cache", args)
        session.close()

    def test_quantized_value_cannot_bypass_flash_attention_validation(self):
        with self.assertRaises(UnsafeRuntimeOptions):
            self.factory().load(self.model, INITIAL_STACK,
                                {"type_v": "Q8_0", "flash_attn": False})
        with self.assertRaises(UnsafeRuntimeOptions):
            self.factory().load(self.model, INITIAL_STACK,
                                {"type_v": "Q8_0"})
        with self.assertRaises(UnsafeRuntimeOptions):
            self.factory().load(self.model, INITIAL_STACK,
                                {"flash_attn": "auto"})
        self.assertEqual(FakeLlama.instances, [])

    def test_session_closed_prevents_generation(self):
        session = self.factory().load(self.model, INITIAL_STACK, {})
        session.close()
        with self.assertRaises(UnverifiedCudaEngine):
            session.generate("test")
        self.assertEqual(FakeLlama.instances[-1].inferences, [])

    def test_session_disallows_public_set_cache_method(self):
        session = self.factory().load(self.model, INITIAL_STACK, {})
        self.assertFalse(hasattr(session, "set_cache"))
        self.assertFalse(hasattr(session, "LlamaDiskCache"))
        self.assertFalse(hasattr(session, "cache_dir"))
        session.close()


if __name__ == "__main__":
    unittest.main()
