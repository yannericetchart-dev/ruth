"""CPU-only tests for read-only GGUF layer analysis and future engine registry."""

import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.layer_matrix import analyze_layer_matrix
from calibration.profile_identity import validate_choices
from calibration.profile_pipeline import Journal, main, run_pipeline
from inference.engine_catalog import ENGINE_CATALOG, engine_info


def tensor(name, byte_count, block, group="feed_forward"):
    return {
        "name": name, "bytes": byte_count,
        "block": block, "group": group, "type": "Q5_K",
        "shape": [16, 32],
    }


class EngineCatalogTests(unittest.TestCase):
    def test_all_user_specified_engines_registered_with_distinct_capabilities(self):
        self.assertEqual(
            set(ENGINE_CATALOG),
            {"llama.cpp", "exllamav2", "vllm", "aphrodite", "tensorrt-llm"}
        )
        self.assertTrue(engine_info("llama.cpp")["model_weight_layer_cpu_offload"])
        self.assertFalse(engine_info("llama.cpp")["kv_cache_host_offload"])
        self.assertEqual(engine_info("llama.cpp")["model_formats"], ("GGUF",))
        for engine in ("exllamav2", "vllm", "aphrodite", "tensorrt-llm"):
            self.assertEqual(engine_info(engine)["status"], "da fare")
            self.assertFalse(engine_info(engine)["enabled"])
            self.assertFalse(
                engine_info(engine)["model_weight_layer_cpu_offload"]
            )
            with self.subTest(engine=engine):
                with self.assertRaisesRegex(NotImplementedError, "da fare"):
                    validate_choices(
                        "qwen-2.5-7b-instruct", "bartowski", engine,
                        engine_info(engine)["method"]
                    )
        self.assertFalse(engine_info("exllamav2")["kv_cache_host_offload"])
        for engine in ("vllm", "aphrodite", "tensorrt-llm"):
            self.assertTrue(engine_info(engine)["kv_cache_host_offload"])

    def test_list_engines_does_not_probe_gpu_or_create_profiles(self):
        output = io.StringIO()
        with (patch("hardware.GPUManager.calibration_status") as gpu,
              redirect_stdout(output)):
            self.assertEqual(main(["--list-engines"]), 0)
        gpu.assert_not_called()
        for name in ENGINE_CATALOG:
            self.assertIn(name, output.getvalue())
        self.assertIn("da fare", output.getvalue())

    def test_future_engine_does_not_fall_back_to_llama(self):
        with self.assertRaisesRegex(NotImplementedError, "da fare"):
            validate_choices(
                "qwen-2.5-7b-instruct", "bartowski",
                "vllm", "llama_cpp_python",
            )


class LayerMatrixTests(unittest.TestCase):
    def test_exact_tensor_enumeration_without_any_cpu_offload(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "qwen-Q5_K_L.gguf"
            path.write_bytes(b"mock")
            baseline = {
                "model_id": "qwen-2.5-7b-instruct",
                "native_verified_all_gpu": True,
                "filename": path.name,
                "local_path": str(path),
                "architecture": "qwen2",
                "quantization": "Q5_K_L",
                "source": "bartowski",
                "baseline_probe_n_ctx": 64,
            }
            entries = [
                tensor("token_embd.weight", 512, None, "embedding"),
                tensor("blk.0.attn_q.weight", 100, 0, "attention"),
                tensor("blk.0.ffn_up.weight", 200, 0),
                tensor("blk.1.attn_q.weight", 150, 1, "attention"),
                tensor("blk.1.ffn_up.weight", 250, 1),
                tensor("output.weight", 64, None, "output"),
            ]
            layout = {
                "metadata": {
                    "general.architecture": "qwen2",
                    "qwen2.block_count": 2,
                },
                "weights_bytes": sum(item["bytes"] for item in entries),
                "tensors": entries,
            }
            with (patch("calibration.layer_matrix.inspect_gguf",
                        return_value=layout) as read,
                  patch("calibration.layer_matrix.engine_info",
                        wraps=engine_info),
                  patch("calibration.layer_matrix.gguf_file_set",
                        return_value=[path])):
                report = analyze_layer_matrix(
                    baseline, model_id="qwen-2.5-7b-instruct"
                )
            read.assert_called_once_with(path, n_ctx=64)
            self.assertTrue(report["success"])
            self.assertEqual(report["status"], "enumeration_only")
            self.assertEqual(report["block_count"], 2)
            self.assertEqual(report["tensor_count"], 6)
            self.assertEqual(
                [r["serialized_weight_bytes"] for r in report["layers"]],
                [300, 400],
            )
            self.assertEqual(report["non_block_serialized_weight_bytes"], 576)
            self.assertEqual(
                report["serialized_weight_bytes"], 1276,
            )
            self.assertEqual(
                report["cpu_prefix_scenarios"][-1]["serialized_block_weight_bytes"],
                700,
            )
            self.assertTrue(all(
                scenario["not_predicted_vram_savings"]
                and not scenario["executable_mapping"]
                for scenario in report["cpu_prefix_scenarios"]
            ))
            self.assertFalse(report["cpu_offload_executed"])
            self.assertFalse(report["gpu_allocated_by_this_phase"])

    def test_matrix_fails_closed_on_bad_baseline_or_missing_block(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.gguf"
            path.write_bytes(b"data")
            baseline = {
                "model_id": "qwen-2.5-7b-instruct",
                "native_verified_all_gpu": False,
                "filename": path.name,
                "local_path": str(path),
                "architecture": "qwen2",
            }
            with patch("calibration.layer_matrix.inspect_gguf") as read:
                with self.assertRaisesRegex(RuntimeError, "non verificata"):
                    analyze_layer_matrix(
                        baseline, model_id="qwen-2.5-7b-instruct"
                    )
                read.assert_not_called()
            baseline["native_verified_all_gpu"] = True
            with patch("calibration.layer_matrix.inspect_gguf",
                       return_value={
                           "metadata": {
                               "general.architecture": "qwen2",
                               "qwen2.block_count": 2,
                           },
                           "weights_bytes": 100,
                           "tensors": [tensor("blk.0.attn_q.weight", 100, 0)],
                       }):
                with self.assertRaisesRegex(RuntimeError, "incompleto"):
                    analyze_layer_matrix(
                        baseline, model_id="qwen-2.5-7b-instruct"
                    )

    def test_matrix_read_only_pipeline_bypasses_context_gpu(self):
        from tests.test_profile_pipeline import identity
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "baseline.gguf"
            path.write_bytes(b"mock")
            ident = identity("qwen-2.5-7b-instruct")

            def choose(**kwargs):
                if kwargs["inventory_only"]:
                    return {
                        "inventory_only": True,
                        "inventory": [{"filename": path.name}],
                        "unranked_quantizations": [],
                    }
                return {
                    "success": True,
                    "attempts": [{}],
                    "baseline": {
                        "model_id": "qwen-2.5-7b-instruct",
                        "source": "bartowski",
                        "repository": "bartowski/Qwen2.5-7B-Instruct-GGUF",
                        "native_verified_all_gpu": True,
                        "filename": path.name,
                        "local_path": str(path),
                        "architecture": "qwen2",
                        "quantization": "Q5_K_L",
                    },
                }

            def layer_matrix(baseline, **kwargs):
                return {
                    "success": True, "status": "enumeration_only",
                    "model_path": str(path), "filename": path.name,
                    "block_count": 2, "tensor_count": 12,
                    "serialized_weight_mib": 2.0,
                    "serialized_weight_bytes": 2097152,
                    "layers": [{}, {}],
                    "cpu_prefix_scenarios": [],
                    "cpu_offload_executed": False,
                }

            with (patch("calibration.baseline_discovery.choose_baseline",
                        side_effect=choose) as selection,
                  patch("calibration.layer_matrix.analyze_layer_matrix",
                        side_effect=layer_matrix) as matrix,
                  patch("calibration.gpu_context.search_gpu_context") as context,
                  redirect_stdout(io.StringIO())):
                report = run_pipeline(ident, until="layer_matrix",
                                      output_root=root)
                repeated = run_pipeline(ident, until="layer_matrix",
                                        output_root=root)
            self.assertTrue(report["success"])
            self.assertTrue(repeated["success"])
            self.assertEqual(selection.call_count, 2)
            self.assertEqual(matrix.call_count, 1)
            context.assert_not_called()
            self.assertTrue(
                (Path(report["run_directory"]) / "layer_matrix.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
