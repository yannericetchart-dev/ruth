"""Verifiche pure Python: non simulano falsi test hardware."""
import unittest

from ruth_runtime import (
    CATALOG, INITIAL_STACKS, Capability, ComponentKind, DevelopmentState,
    StackSelection, UnsupportedStackError, components, inspect_stack,
)
from ruth_runtime.capabilities import PLATFORM_OS, PLATFORM_ARCH


class CatalogContractTests(unittest.TestCase):
    def test_platform_is_explicit(self):
        self.assertEqual((PLATFORM_OS, PLATFORM_ARCH), ("GNU/Linux", "x86_64"))

    def test_catalog_has_unique_keys_and_status(self):
        keys = [(c.kind, c.identifier) for c in CATALOG]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(isinstance(c, Capability) for c in CATALOG))
        self.assertFalse(any(c.state == DevelopmentState.VERIFIED for c in CATALOG))
        self.assertFalse(any(c.state == DevelopmentState.IMPLEMENTED for c in CATALOG))

    def test_expected_first_targets(self):
        for kind, name in (
            (ComponentKind.CPU_VENDOR, "intel"),
            (ComponentKind.GPU_VENDOR, "nvidia"),
            (ComponentKind.GPU_BACKEND, "cuda"),
            (ComponentKind.NATIVE_LIBRARY, "ggml-cpu"),
            (ComponentKind.INFERENCE_ENGINE, "llama.cpp"),
            (ComponentKind.INITIALIZATION, "llama_cpp_python"),
        ):
            matches = [c for c in components(kind) if c.identifier == name]
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].state, DevelopmentState.TARGET)

    def test_todo_vendors_backends_engines_and_initialization_modes(self):
        required = {
            ComponentKind.CPU_VENDOR: ("amd",),
            ComponentKind.GPU_VENDOR: ("amd", "intel"),
            ComponentKind.GPU_BACKEND: ("vulkan", "rocm", "hip", "sycl", "oneapi", "xpu", "openvino-gpu"),
            ComponentKind.CPU_BACKEND: ("onednn", "onemkl", "openvino-cpu", "onnxruntime-cpu", "vllm-cpu", "blis", "openblas"),
            ComponentKind.INFERENCE_ENGINE: ("vllm", "onnxruntime", "pytorch", "openvino", "mlc-llm", "iree", "sglang"),
            ComponentKind.INITIALIZATION: ("local-server", "system-library", "cli"),
        }
        for kind, names in required.items():
            values = {c.identifier: c.state for c in components(kind)}
            for name in names:
                self.assertEqual(values[name], DevelopmentState.TO_DO, (kind, name))

    def test_target_stacks_not_marked_runnable(self):
        self.assertEqual(len(INITIAL_STACKS), 1)
        self.assertEqual(INITIAL_STACKS[0], StackSelection(
            "gpu", "nvidia", "cuda", "llama.cpp", "llama_cpp_python"
        ))
        for stack in INITIAL_STACKS:
            assessment = inspect_stack(stack)
            self.assertEqual(assessment.state, DevelopmentState.TARGET)
            self.assertFalse(assessment.runnable)

    def test_ggml_cpu_is_library_not_selectable_backend(self):
        self.assertEqual([c.identifier for c in components(ComponentKind.NATIVE_LIBRARY)], ["ggml-cpu"])
        self.assertNotIn("ggml-cpu", [c.identifier for c in components(ComponentKind.CPU_BACKEND)])
        with self.assertRaises(UnsupportedStackError):
            inspect_stack(StackSelection("cpu", "intel", "ggml-cpu", "llama.cpp", "llama_cpp_python"))
        # Tecnologie CPU future possono essere censite, mai promosse a TARGET di runtime.
        future = inspect_stack(StackSelection("cpu", "intel", "onednn", "llama.cpp", "llama_cpp_python"))
        self.assertEqual(future.state, DevelopmentState.TO_DO)
        self.assertFalse(future.runnable)

    def test_cross_vendor_future_stack_not_marked_runnable(self):
        stack = StackSelection("gpu", "amd", "rocm", "llama.cpp", "cli")
        assessment = inspect_stack(stack)
        self.assertEqual(assessment.state, DevelopmentState.TO_DO)
        self.assertFalse(assessment.runnable)

    def test_unknown_component_and_cross_kind_rejected(self):
        bad = (
            StackSelection("gpu", "nvidia", "nonexistent", "llama.cpp", "llama_cpp_python"),
            StackSelection("cpu", "intel", "cuda", "llama.cpp", "llama_cpp_python"),
            StackSelection("asic", "nvidia", "cuda", "llama.cpp", "llama_cpp_python"),
        )
        for stack in bad:
            with self.subTest(stack=stack):
                with self.assertRaises(UnsupportedStackError):
                    inspect_stack(stack)


if __name__ == "__main__":
    unittest.main()
