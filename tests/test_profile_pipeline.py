"""CPU-only unit tests for Ruth's incremental user/hardware calibration workflow."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.profile_identity import resolve_identity
from calibration.profile_pipeline import (
    IMPLEMENTED, ROADMAP, Journal, brief_event, main, run_pipeline,
)


GPU_STATUS = {
    "ready": True,
    "errors": [],
    "foreign_processes": [],
    "gpu": {
        "vendor": "nvidia", "name": "Mock RTX 4050",
        "uuid": "GPU-test-4050", "index": 0,
    },
    "memory": {"total_mib": 6141.0},
}


def identity(model="gemma-2-9b-it", gpu=GPU_STATUS):
    with (
        patch("calibration.profile_identity._ram_total_mib", return_value=32000),
        patch("calibration.profile_identity._cpu_model", return_value="Test CPU"),
        patch("calibration.profile_identity._installed", return_value="0.3.36"),
    ):
        return resolve_identity(model, status=gpu)


class ProfileIdentityTests(unittest.TestCase):
    def test_source_engine_method_and_backend_are_distinct(self):
        data = identity("llama-3.1-8b-instruct")
        self.assertEqual(data["choices"]["source"], "bartowski")
        self.assertEqual(data["choices"]["engine"], "llama.cpp")
        self.assertEqual(data["choices"]["method"], "llama_cpp_python")
        self.assertEqual(data["hardware"]["backend"], "cuda")
        self.assertEqual(data["hardware"]["vram_total_mib"], 6141.0)
        self.assertEqual(data["hardware"]["ram_total_mib"], 32000)
        self.assertEqual(data["software"]["llama_cpp_python"], "0.3.36")

    def test_profile_id_is_stable_but_differs_by_model_or_gpu(self):
        a = identity()
        self.assertEqual(a["profile_key"], identity()["profile_key"])
        self.assertNotEqual(
            a["profile_key"], identity("llama-3.1-8b-instruct")["profile_key"]
        )
        gpu = {
            **GPU_STATUS, "gpu": {**GPU_STATUS["gpu"], "uuid": "GPU-other"},
        }
        self.assertNotEqual(a["profile_key"], identity(gpu=gpu)["profile_key"])
        changed = {
            **GPU_STATUS,
            "memory": {"total_mib": 12282.0},
        }
        self.assertNotEqual(a["profile_key"], identity(gpu=changed)["profile_key"])

    def test_unsupported_choices_do_not_silently_fallback(self):
        with self.assertRaisesRegex(ValueError, "Fonte"):
            resolve_identity("gemma-2-9b-it", source="other", status=GPU_STATUS)
        with self.assertRaisesRegex(ValueError, "Motore/metodo"):
            resolve_identity(
                "gemma-2-9b-it", engine="onnx", status=GPU_STATUS
            )
        unsupported = {
            **GPU_STATUS, "gpu": {**GPU_STATUS["gpu"], "vendor": "amd"},
        }
        with self.assertRaisesRegex(RuntimeError, "Backend"):
            resolve_identity("gemma-2-9b-it", status=unsupported)

    def test_kaggle_and_hf_choices_have_distinct_profiles(self):
        with (
            patch("calibration.profile_identity._ram_total_mib",
                  return_value=32000),
            patch("calibration.profile_identity._cpu_model",
                  return_value="Test CPU"),
            patch("calibration.profile_identity._installed",
                  return_value="0.3.36"),
        ):
            for model, handle in (
                ("gemma-2-9b-it", "google/gemma-2/gguf/2.0-9b-it"),
                ("qwen-2.5-7b-instruct",
                 "qwen-lm/qwen2.5/gguf/7b-instruct"),
            ):
                with self.subTest(model=model):
                    hf = resolve_identity(model, status=GPU_STATUS)
                    kaggle = resolve_identity(
                        model, source="kaggle", status=GPU_STATUS
                    )
                    self.assertNotEqual(hf["profile_key"],
                                        kaggle["profile_key"])
                    self.assertEqual(kaggle["choices"]["kaggle_handle"],
                                     handle)
                    self.assertEqual(hf["choices"]["source"], "bartowski")
            with self.assertRaisesRegex(ValueError, "Fonte"):
                resolve_identity("mistral-7b-instruct-v0.3",
                                 source="kaggle", status=GPU_STATUS)

    def test_source_specific_model_directories_are_isolated(self):
        from model import ModelManager
        with tempfile.TemporaryDirectory() as temp:
            manager = ModelManager(root=temp)
            model = "gemma-2-9b-it"
            hf_dir = manager.model_directory(model, "bartowski")
            kaggle_dir = manager.model_directory(model, "kaggle")
            self.assertEqual(hf_dir, Path(temp) / model)
            self.assertEqual(kaggle_dir, Path(temp) / model /
                             "sources" / "kaggle" / "google" / "gemma-2")
            self.assertNotEqual(kaggle_dir, hf_dir)
            alternate = manager.model_directory(
                model, "kaggle", "otherauthor/othermodel"
            )
            self.assertNotEqual(alternate, kaggle_dir)

    def test_unready_gpu_blocks_profiling(self):
        with self.assertRaisesRegex(RuntimeError, "GPU non pronta"):
            resolve_identity("gemma-2-9b-it", status={
                **GPU_STATUS, "ready": False, "errors": ["Other GPU process"],
            })


class ProfilePipelineTests(unittest.TestCase):
    def test_roadmap_lists_future_without_running_it(self):
        self.assertEqual(IMPLEMENTED, ["inventory", "baseline", "context_gpu"])
        self.assertEqual(len(ROADMAP), 7)
        self.assertFalse(dict(ROADMAP)["mapping"])
        self.assertIn("Test n_ctx=128", brief_event({
            "event": "context_probe_start", "n_ctx": 128, "token_count": 8,
        }))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "non implementata"):
                run_pipeline(identity(), until="mapping", output_root=Path(temp))

    def test_full_run_persists_per_configuration_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "Q4_K_M.gguf"
            model.write_bytes(b"mock")
            calls = []
            def fake_choose(**kwargs):
                self.assertFalse(kwargs["persist_selection"])
                if kwargs["inventory_only"]:
                    calls.append("inventory")
                    kwargs["progress"]({
                        "event": "inventory_ready", "model": kwargs["model_id"],
                        "count": 2, "unranked": 0,
                    })
                    return {
                        "inventory_only": True, "inventory": [
                            {"filename": "Q5_K_L.gguf"}, {"filename": "Q4_K_M.gguf"},
                        ], "unranked_quantizations": [],
                    }
                calls.append("baseline")
                kwargs["progress"]({
                    "event": "candidate_start", "model": kwargs["model_id"],
                    "filename": model.name, "quantization": "Q4_K_M",
                })
                kwargs["progress"]({
                    "event": "candidate_decision",
                    "model": kwargs["model_id"],
                    "quantization": "Q4_K_M",
                    "status": "selected_full_gpu_baseline",
                    "peak_mib": 6107.3,
                })
                return {
                    "success": True, "attempts": [{}],
                    "baseline": {
                        "model_id": kwargs["model_id"],
                        "repository": kwargs["repo_id"],
                        "architecture": "gemma2",
                        "filename": model.name,
                        "local_path": str(model),
                        "native_verified_all_gpu": True,
                        "quantization": "Q4_K_M",
                    },
                }

            def fake_ctx(model_id, **kwargs):
                self.assertFalse(kwargs["persist_reference"])
                self.assertEqual(kwargs["baseline_selection"]["filename"], model.name)
                calls.append("context_gpu")
                for ctx in (64, 256, 512, 384, 257):
                    kwargs["progress"]({
                        "event": "context_probe_start", "n_ctx": ctx,
                        "token_count": 8, "model": model_id,
                    })
                    kwargs["progress"]({
                        "event": "context_probe_result", "n_ctx": ctx,
                        "model": model_id, "success": ctx <= 256,
                        "failure_class": None if ctx <= 256 else "native_cuda_oom",
                        "peak_mib": 6107.3 if ctx <= 256 else None,
                        "native_gpu_weights": ctx <= 256,
                        "native_gpu_kv": ctx <= 256,
                    })
                return {
                    "success": True,
                    "confirmed_max_gpu_resident_n_ctx": 256,
                    "confirmation_min_sampled_headroom_mib": 33.7,
                    "attempts": [{}] * 5,
                    "profile_generated": False,
                }

            config = identity()
            log = io.StringIO()
            with (
                patch("calibration.baseline_discovery.choose_baseline",
                      side_effect=fake_choose),
                patch("calibration.gpu_context.search_gpu_context",
                      side_effect=fake_ctx),
                redirect_stdout(log),
            ):
                first = run_pipeline(config, output_root=root)
                second = run_pipeline(config, output_root=root)
            self.assertTrue(first["success"])
            self.assertTrue(second["success"])
            self.assertEqual(calls, ["inventory", "baseline", "context_gpu"])
            self.assertIn("Test n_ctx=256", log.getvalue())
            self.assertIn("Decisione: Q4_K_M", log.getvalue())
            self.assertIn("RIUTILIZZO", log.getvalue())
            directory = Path(first["run_directory"])
            self.assertEqual(directory.name, config["profile_key"])
            state = json.loads((directory / "state.json").read_text())
            self.assertEqual(
                [state["phases"][p]["status"] for p in IMPLEMENTED],
                ["completed"] * 3,
            )
            self.assertFalse(state["profile_generated"])
            self.assertFalse(state["operationally_selected"])
            self.assertEqual(
                json.loads((directory / "context_gpu.json").read_text())[
                    "confirmed_max_gpu_resident_n_ctx"], 256
            )
            self.assertIn("context_probe_result",
                          (directory / "events.jsonl").read_text())
            self.assertFalse((model.parent / "baseline_selection.json").exists())
            self.assertFalse((model.parent / "gpu_context_reference.json").exists())

    def test_failed_context_is_blocked_then_retry_without_repeating_prior(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "Q5_K_L.gguf"
            model.write_bytes(b"mock")
            runs = []
            def fake_choose(**kwargs):
                if kwargs["inventory_only"]:
                    runs.append("inventory")
                    return {"inventory_only": True, "inventory": [{"filename": "x"}],
                            "unranked_quantizations": []}
                runs.append("baseline")
                return {
                    "success": True, "attempts": [{}],
                    "baseline": {
                        "model_id": "qwen-2.5-7b-instruct",
                        "repository": "bartowski/Qwen2.5-7B-Instruct-GGUF",
                        "filename": model.name, "local_path": str(model),
                        "native_verified_all_gpu": True, "quantization": "Q5_K_L",
                    },
                }

            def fake_context(*args, **kwargs):
                runs.append("context")
                if runs.count("context") == 1:
                    return {
                        "success": False, "confirmed_max_gpu_resident_n_ctx": None,
                        "error": "JSON worker non valido",
                        "attempts": [{}] * 6,
                    }
                return {
                    "success": True, "confirmed_max_gpu_resident_n_ctx": 512,
                    "attempts": [{}] * 8,
                }

            config = identity("qwen-2.5-7b-instruct")
            with (
                patch("calibration.baseline_discovery.choose_baseline",
                      side_effect=fake_choose),
                patch("calibration.gpu_context.search_gpu_context",
                      side_effect=fake_context),
                redirect_stdout(io.StringIO()),
            ):
                first = run_pipeline(config, output_root=root)
                second = run_pipeline(config, output_root=root)
            self.assertFalse(first["success"])
            self.assertEqual(first["blocked_phase"], "context_gpu")
            self.assertTrue(second["success"])
            self.assertEqual(runs, ["inventory", "baseline", "context", "context"])
            state = json.loads((Path(second["run_directory"]) / "state.json").read_text())
            self.assertEqual(state["phases"]["context_gpu"]["status"], "completed")

    def test_inventory_unranked_blocks_baseline_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            config = identity()
            with (
                patch("calibration.baseline_discovery.choose_baseline",
                      return_value={
                          "inventory_only": True, "inventory": [
                              {"filename": "unknown.gguf"}
                          ],
                          "unranked_quantizations": ["unknown.gguf"],
                      }),
                redirect_stdout(io.StringIO()),
            ):
                outcome = run_pipeline(
                    config, until="baseline", output_root=Path(temp)
                )
            self.assertFalse(outcome["success"])
            self.assertEqual(outcome["blocked_phase"], "inventory")

    def test_upstream_rerun_marks_downstream_results_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = Journal(root / "run", identity())
            for phase in IMPLEMENTED:
                (journal.folder / (phase + ".json")).write_text(json.dumps({
                    "success": True, "inventory_only": True,
                    "unranked_quantizations": [],
                    "confirmed_max_gpu_resident_n_ctx": 256,
                }))
                journal.phase_state(phase, "completed", phase + ".json")
            journal.invalidate_after("inventory")
            self.assertEqual(
                journal.state["phases"]["inventory"]["status"], "completed"
            )
            self.assertEqual(
                journal.state["phases"]["baseline"]["status"], "stale"
            )
            self.assertEqual(
                journal.state["phases"]["context_gpu"]["status"], "stale"
            )
            self.assertIsNone(journal.cached("context_gpu"))
            # Old diagnostic report remains readable without being current.
            self.assertTrue((journal.folder / "context_gpu.json").is_file())
            self.assertEqual(
                json.loads((journal.folder / "state.json").read_text())[
                    "phases"]["baseline"]["stale_due_to"], "inventory"
            )

    def test_source_filter_all_selects_two_kaggle_and_four_hf_models(self):
        recorded = []
        def fake_identity(model, **kwargs):
            recorded.append((model, kwargs.get("source")))
            return {"profile_key": model}
        with (
            patch("calibration.profile_pipeline.resolve_identity",
                  side_effect=fake_identity),
            patch("calibration.profile_pipeline.run_pipeline",
                  return_value={"success": True}),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(["--model", "all", "--source",
                                   "kaggle", "--until", "inventory"]), 0)
            self.assertEqual(
                {m for m, _ in recorded},
                {"gemma-2-9b-it", "qwen-2.5-7b-instruct"},
            )
            recorded.clear()
            self.assertEqual(main(["--model", "all", "--source",
                                   "bartowski", "--until", "inventory"]), 0)
            self.assertEqual(len(recorded), 4)
            self.assertIn(("mistral-7b-instruct-v0.3", "bartowski"),
                          recorded)

    def test_cli_lists_registered_phases_without_gpu(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--list-phases"]), 0)
        self.assertIn("mapping", out.getvalue())
        self.assertIn("previsto", out.getvalue())


if __name__ == "__main__":
    unittest.main()
