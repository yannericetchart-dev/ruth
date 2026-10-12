"""Unit coverage for five-pair phase 1 auditing, no GPU model loads."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.phase1_audit import audit_pair, model_source_pairs
from model import ModelManager


class PhaseOneAuditTests(unittest.TestCase):
    def test_registry_has_exactly_five_model_source_pairs(self):
        pairs = model_source_pairs()
        self.assertEqual(len(pairs), 5)
        self.assertEqual(
            len(model_source_pairs(source="bartowski")), 4
        )
        self.assertEqual(
            model_source_pairs(source="kaggle"),
            [("qwen-2.5-7b-instruct", "kaggle")]
        )

    def identity(self, model="qwen-2.5-7b-instruct",
                 source="bartowski"):
        spec = ModelManager.source_spec(model, source)
        return {
            "profile_key": "synthetic-identity-audit",
            "choices": {
                "model_id": model, "source": source,
                "repository": spec["repository"],
                "kaggle_handle": spec.get("kaggle_handle"),
            },
            "reference_runtime": {
                "context_confirm_rounds": 2,
            },
        }

    def prepare(self, root, *, with_context=True):
        manager = ModelManager(root=root / "models")
        ident = self.identity()
        model = "qwen-2.5-7b-instruct"
        spec = ModelManager.source_spec(model, "bartowski")
        weights = manager.model_directory(model, "bartowski")
        weights.mkdir(parents=True)
        gguf = weights / "selected.gguf"
        gguf.write_bytes(b"simulated weights")
        run = root / "runs" / ident["profile_key"]
        run.mkdir(parents=True)
        baseline_selected = {
            "model_id": model, "source": "bartowski",
            "repository": spec["repository"], "architecture": "qwen2",
            "filename": gguf.name, "local_path": str(gguf),
            "quantization": "Q4_K_M", "native_verified_all_gpu": True,
            "baseline_probe_n_ctx": 64,
        }
        baseline = {
            "success": True,
            "baseline": baseline_selected,
            "attempts": [{
                "status": "selected_full_gpu_baseline",
                "filename": gguf.name,
                "test": {
                    "success": True,
                    "full_gpu_native_verification": {"verified": True},
                    "measurements": {
                        "minimal_inference_completed": True,
                        "vram_peak_sampled_mib": 5900,
                    },
                },
            }],
        }
        def good_probe():
            return {
                "requested_n_ctx": 256, "success": True,
                "full_gpu_native_verification": {"verified": True},
                "gpu_runtime_validation": {
                    "verified": True,
                    "kv": {"verified_gpu_resident": True},
                },
            }
        context = {
            "success": True, "measurement_completed": True,
            "model_id": model, "filename": gguf.name,
            "quantization": "Q4_K_M",
            "confirmed_max_gpu_resident_n_ctx": 256,
            "confirmation": [{"success": True}, {"success": True}],
            "attempts": [good_probe(), good_probe()],
            "confirmation_min_sampled_headroom_mib": 50,
            "search_ceiling": 8192,
        }
        files = {
            "inventory": {"inventory_only": True,
                          "inventory": [{"filename": gguf.name}],
                          "unranked_quantizations": []},
            "baseline": baseline,
        }
        if with_context:
            files["context_gpu"] = context
        phases = {}
        for key, data in files.items():
            (run / (key + ".json")).write_text(json.dumps(data))
            phases[key] = {
                "status": "completed",
                "result_file": key + ".json"
            }
        state = {"identity": ident, "phases": phases}
        (run / "state.json").write_text(json.dumps(state))
        return manager, ident, run, gguf

    def test_complete_profile_requires_cuda_verified_baseline_and_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager, ident, run, gguf = self.prepare(root)
            state_before = (run / "state.json").read_bytes()
            layout = {
                "metadata": {"general.architecture": "qwen2"},
                "shard_count": 1, "tensor_count": 339,
                "weights_bytes": 10000,
            }
            real_manager = ModelManager
            with (
                patch("calibration.phase1_audit.ModelManager",
                      side_effect=lambda: real_manager(root=manager.root)),
                patch("calibration.phase1_audit.inspect_gguf",
                      return_value=layout) as inspection,
            ):
                result = audit_pair(ident, output_root=root / "runs")
            inspection.assert_called_once()
            self.assertTrue(result["phase1_complete"])
            self.assertTrue(result["downloaded_local_gguf_verified"])
            self.assertTrue(result["full_gpu_baseline_verified"])
            self.assertTrue(result["context_gpu_resident_verified"])
            self.assertEqual(result["n_ctx_gpu_resident"], 256)
            self.assertEqual(result["checks"]["tensor_count"], 339)
            self.assertEqual((run / "state.json").read_bytes(), state_before)

    def test_blocked_context_preserves_verified_floor_but_not_max(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager, ident, run, gguf = self.prepare(root)
            context_file = run / "context_gpu.json"
            data = json.loads(context_file.read_text())
            data.update({
                "success": False, "measurement_completed": False,
                "confirmation_completed": True,
                "maximality_proven": False,
                "confirmed_max_gpu_resident_n_ctx": None,
                "confirmed_gpu_resident_n_ctx": 256,
                "uncertain_failure_boundaries": [{
                    "n_ctx": 300, "failure_class": "worker_protocol_error",
                    "signal": "SIGABRT",
                }],
            })
            context_file.write_text(json.dumps(data))
            state_path = run / "state.json"
            state = json.loads(state_path.read_text())
            state["phases"]["context_gpu"]["status"] = "blocked"
            state_path.write_text(json.dumps(state))
            before = state_path.read_bytes()

            with patch(
                "calibration.phase1_audit.ModelManager",
                side_effect=lambda: ModelManager(root=manager.root),
            ):
                result = audit_pair(
                    ident, output_root=root / "runs", inspect_local=False
                )
            self.assertFalse(result["phase1_complete"])
            self.assertFalse(result["context_gpu_resident_verified"])
            self.assertIsNone(result["n_ctx_gpu_resident"])
            self.assertEqual(
                result["confirmed_lower_bound_gpu_resident_n_ctx"], 256
            )
            self.assertFalse(result["context_maximality_proven"])
            self.assertIn("massimo inconclusivo", result["errors"][0])
            self.assertEqual(state_path.read_bytes(), before)

    def test_blocked_context_never_accepts_floor_without_native_proofs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager, ident, run, gguf = self.prepare(root)
            context_file = run / "context_gpu.json"
            data = json.loads(context_file.read_text())
            data.update({
                "success": False, "measurement_completed": False,
                "confirmation_completed": True,
                "maximality_proven": False,
                "confirmed_max_gpu_resident_n_ctx": None,
                "confirmed_gpu_resident_n_ctx": 256,
                "uncertain_failure_boundaries": [{"n_ctx": 300}],
            })
            data["attempts"][0]["gpu_runtime_validation"]["kv"][
                "verified_gpu_resident"
            ] = False
            data["attempts"][1]["gpu_runtime_validation"]["kv"][
                "verified_gpu_resident"
            ] = False
            context_file.write_text(json.dumps(data))
            state_path = run / "state.json"
            state = json.loads(state_path.read_text())
            state["phases"]["context_gpu"]["status"] = "blocked"
            state_path.write_text(json.dumps(state))
            with patch(
                "calibration.phase1_audit.ModelManager",
                side_effect=lambda: ModelManager(root=manager.root),
            ):
                result = audit_pair(
                    ident, output_root=root / "runs", inspect_local=False
                )
            self.assertIsNone(
                result["confirmed_lower_bound_gpu_resident_n_ctx"]
            )
            self.assertFalse(result["phase1_complete"])
            self.assertFalse(result["context_gpu_resident_verified"])

    def test_missing_context_is_incomplete_and_does_not_probe_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager, ident, run, gguf = self.prepare(
                root, with_context=False
            )
            with patch("calibration.phase1_audit.ModelManager",
                       side_effect=lambda: ModelManager(root=manager.root)):
                result = audit_pair(
                    ident, output_root=root / "runs",
                    inspect_local=False,
                )
            self.assertTrue(result["full_gpu_baseline_verified"])
            self.assertFalse(result["phase1_complete"])
            self.assertFalse(result["context_gpu_resident_verified"])
            self.assertIsNone(result["n_ctx_gpu_resident"])
            self.assertIn("non ancora completato", result["errors"][0])

    def test_local_matrix_uses_verified_download_without_remote_queries(self):
        from calibration.layer_matrix import local_baseline_matrices
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "runs" / "saved_qwen"
            folder.mkdir(parents=True)
            baseline = {
                "baseline": {
                    "model_id": "qwen-2.5-7b-instruct",
                    "native_verified_all_gpu": True,
                    "filename": "chosen-Q5_K_L.gguf",
                    "quantization": "Q5_K_L",
                }
            }
            (folder / "baseline.json").write_text(json.dumps(baseline))
            scanned = [{
                "model_id": "qwen-2.5-7b-instruct",
                "source": "bartowski",
                "run_directory": str(folder),
                "full_gpu_baseline_verified": True,
                "errors": [],
            }]
            report = {
                "schema_version": 2,
                "block_count": 28, "tensor_count": 339,
                "quantization": "Q5_K_L",
                "serialized_weight_mib": 5507.7,
            }
            with (
                patch("calibration.phase1_audit.audit_all",
                      return_value=scanned) as audit,
                patch("calibration.layer_matrix.analyze_layer_matrix",
                      return_value=report) as local,
                patch("calibration.layer_export.export_matrix_tables",
                      return_value={}) as csv,
            ):
                result = local_baseline_matrices(
                    ["qwen-2.5-7b-instruct"],
                    source="bartowski", output_root=root / "runs",
                )
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["quantization"], "Q5_K_L")
            self.assertEqual(local.call_count, 1)
            self.assertEqual(csv.call_count, 1)
            self.assertEqual(audit.call_count, 1)
            self.assertTrue(Path(result[0]["report_path"]).is_file())

    def test_missing_shard_or_empty_local_file_is_not_reported_as_downloaded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager, ident, run, gguf = self.prepare(root)
            gguf.write_bytes(b"")
            with patch("calibration.phase1_audit.ModelManager",
                       side_effect=lambda: ModelManager(root=manager.root)):
                result = audit_pair(ident, output_root=root / "runs")
            self.assertFalse(result["downloaded_local_gguf_verified"])
            self.assertFalse(result["full_gpu_baseline_verified"])
            self.assertFalse(result["phase1_complete"])


if __name__ == "__main__":
    unittest.main()
