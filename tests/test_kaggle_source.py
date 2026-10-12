"""Kaggle GGUF source compatibility without network or multi-GB downloads."""

import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from calibration.baseline_discovery import minimal_chat_prompt, choose_baseline
from calibration.kaggle_source import (
    inventory_kaggle, validate_handle, download_kaggle,
)
from calibration.profile_identity import resolve_identity
from calibration.profile_pipeline import main


REPOSITORY = "qwen-lm/qwen2.5"
HANDLE = REPOSITORY + "/gguf/7b-instruct/1"


def mock_kaggle_call(argv, **kwargs):
    if argv[2:4] == ["variations", "list"]:
        return subprocess.CompletedProcess(
            argv, 0,
            "Framework,Variation\nGGUF,quantizations\n", ""
        )
    if (argv[2:4] == ["variations", "files"] or
        argv[2:5] == ["variations", "versions", "files"]):
        return subprocess.CompletedProcess(
            argv, 0,
            "Name,Size\n"
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf,4.2GB\n"
            "Qwen2.5-7B-Instruct-Q5_K_M.gguf,5.2GB\n"
            "README.md,10KB\n", ""
        )
    raise AssertionError(f"Unexpected external command: {argv}")


class KaggleSourceTests(unittest.TestCase):
    def test_cli_uses_virtualenv_executable_without_activation(self):
        from calibration.kaggle_source import _csv_call

        with tempfile.TemporaryDirectory() as tmp:
            fake_python = Path(tmp) / "python"
            fake_cli = Path(tmp) / "kaggle"
            fake_cli.write_text("#!/bin/sh\n")
            with (
                patch("calibration.kaggle_source.sys.executable",
                      str(fake_python)),
                patch("calibration.kaggle_source.subprocess.run",
                      return_value=subprocess.CompletedProcess(
                          [], 0, "Name,Size\n", ""
                      )) as call,
            ):
                self.assertEqual(
                    _csv_call("variations", "files", HANDLE),
                    [],
                )
            self.assertEqual(call.call_args.args[0][0], str(fake_cli))

    def test_missing_handle_does_not_call_unsupported_cli(self):
        with patch("calibration.kaggle_source.subprocess.run") as cmd:
            with self.assertRaisesRegex(RuntimeError, "non supporta"):
                inventory_kaggle(
                    REPOSITORY, "Qwen2.5-7B-Instruct-"
                )
        cmd.assert_not_called()

    def test_versioned_handle_uses_version_files_subcommand(self):
        with patch("calibration.kaggle_source.subprocess.run",
                   side_effect=mock_kaggle_call) as call:
            result = inventory_kaggle(
                REPOSITORY, "Qwen2.5-7B-Instruct-",
                selected_handle=HANDLE,
            )
        self.assertTrue(result)
        self.assertEqual(call.call_args.args[0][2:5],
                         ["variations", "versions", "files"])

    def test_explicit_pinned_handle_bypasses_variation_discovery(self):
        with patch("calibration.kaggle_source.subprocess.run",
                   side_effect=mock_kaggle_call) as call:
            result = inventory_kaggle(
                REPOSITORY, "Qwen2.5-7B-Instruct-",
                selected_handle=HANDLE,
            )
        self.assertEqual(call.call_count, 1)
        self.assertTrue(all(x["kaggle_handle"] == HANDLE for x in result))

    def test_missing_gguf_stops_without_hf_fallback(self):
        only_pytorch = subprocess.CompletedProcess(
            [], 0, "Name\nconsolidated.safetensors\n", ""
        )
        with (
            patch("calibration.kaggle_source.subprocess.run",
                  return_value=only_pytorch),
            patch("huggingface_hub.hf_hub_download") as hf,
        ):
            with self.assertRaisesRegex(RuntimeError, "Nessun file .gguf"):
                inventory_kaggle(
                    REPOSITORY, "Qwen2.5-7B-Instruct-",
                    selected_handle=HANDLE,
                )
            hf.assert_not_called()

    def test_handle_validation_never_accepts_foreign_model(self):
        self.assertEqual(validate_handle(HANDLE, REPOSITORY), HANDLE)
        for invalid in (
            "bond005/qwen-2.5-7b-instruct",
            "other/qwen-2.5-7b-instruct/gguf/quantizations",
            "../etc/passwd/gguf/quantizations",
            REPOSITORY + "/gguf/../../bad",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_handle(invalid, REPOSITORY)

    def test_single_gguf_download_uses_kagglehub_cache_not_hf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "cache" / "Qwen2.5-7B-Instruct-Q5_K_M.gguf"
            source.parent.mkdir()
            source.write_bytes(b"GGUFfake")
            candidate = {
                "filename": source.name, "source_file_path": source.name,
                "repo_id": REPOSITORY, "kaggle_handle": HANDLE,
            }
            with (
                patch("kagglehub.model_download", return_value=str(source)) as kaggle,
                patch("huggingface_hub.hf_hub_download") as hf,
            ):
                result = download_kaggle(candidate, root / "models")
            self.assertEqual(result.read_bytes(), source.read_bytes())
            kaggle.assert_called_once_with(HANDLE, path=source.name)
            hf.assert_not_called()

    def test_mistral_uses_inst_not_llama3_special_header(self):
        prompt = minimal_chat_prompt("llama", "mistral-7b-instruct-v0.3")
        self.assertEqual(prompt, "[INST] Ciao. [/INST]")
        self.assertNotIn("<|start_header_id|>", prompt)
        self.assertIn(
            "<|start_header_id|>",
            minimal_chat_prompt("llama", "llama-3.1-8b-instruct"),
        )

    def test_kaggle_inventory_pipeline_does_not_download(self):
        gpu = {
            "ready": True,
            "gpu": {"vendor": "nvidia", "uuid": "GPU-test",
                    "name": "NVIDIA Test", "index": 0},
            "memory": {"total_mib": 6141},
            "errors": [], "foreign_processes": [],
        }
        with (
            patch("calibration.baseline_discovery.GPUManager.calibration_status",
                  return_value=gpu),
            patch("calibration.kaggle_source.inventory_kaggle",
                  return_value=[
                      {
                          "filename": "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
                          "quantization": "Q5_K_M",
                          "quality_priority": 700,
                      },
                  ]),
            patch("huggingface_hub.hf_hub_download") as hf,
        ):
            result = choose_baseline(
                model_id="qwen-2.5-7b-instruct", source="kaggle",
                inventory_only=True,
            )
        self.assertTrue(result["inventory_only"])
        self.assertEqual(result["source"], "kaggle")
        self.assertEqual(len(result["inventory"]), 1)
        hf.assert_not_called()

    def test_both_official_variations_are_passed_to_kaggle_inventory(self):
        from model import ModelManager
        for model in ("gemma-2-9b-it", "qwen-2.5-7b-instruct"):
            with self.subTest(model=model):
                spec = ModelManager.source_spec(model, "kaggle")
                gpu = {
                    "ready": True, "errors": [], "foreign_processes": [],
                    "gpu": {"vendor": "nvidia", "name": "Mock CUDA",
                            "uuid": "GPU-test", "index": 0},
                    "memory": {"total_mib": 6141},
                }
                with (
                    patch("calibration.baseline_discovery.GPUManager.calibration_status",
                          return_value=gpu),
                    patch("calibration.kaggle_source.inventory_kaggle",
                          return_value=[{
                              "filename": "synthetic-Q4_K_M.gguf",
                              "quantization": "Q4_K_M",
                              "quality_priority": 400,
                          }]) as listed,
                    patch("huggingface_hub.hf_hub_download") as hf,
                ):
                    result = choose_baseline(
                        model_id=model, source="kaggle",
                        inventory_only=True,
                    )
                listed.assert_called_once_with(
                    spec["repository"], spec["gguf_prefix"],
                    selected_handle=spec["kaggle_handle"],
                )
                hf.assert_not_called()
                self.assertEqual(result["repository"], spec["repository"])

    def test_mistral_huggingface_never_queries_kaggle(self):
        from model import ModelManager
        spec = ModelManager.source_spec("mistral-7b-instruct-v0.3")
        self.assertEqual(
            spec["repository"], "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
        )
        gpu = {
            "ready": True, "errors": [], "foreign_processes": [],
            "gpu": {"vendor": "nvidia", "name": "Mock CUDA",
                    "uuid": "GPU-test", "index": 0},
            "memory": {"total_mib": 6141},
        }
        with (
            patch("calibration.baseline_discovery.GPUManager.calibration_status",
                  return_value=gpu),
            patch("calibration.baseline_discovery.inventory",
                  return_value=[{"filename": spec["filename"],
                                 "quantization": "Q4_K_M",
                                 "quality_priority": 400}]),
            patch("calibration.kaggle_source.inventory_kaggle") as kaggle,
        ):
            result = choose_baseline(
                model_id="mistral-7b-instruct-v0.3", inventory_only=True
            )
        self.assertEqual(result["source"], "bartowski")
        kaggle.assert_not_called()

    def test_context_rejects_huggingface_baseline_for_kaggle(self):
        from calibration.gpu_context import search_gpu_context
        gpu = {
            "ready": True, "errors": [], "foreign_processes": [],
            "gpu": {"vendor": "nvidia", "name": "Mock CUDA",
                    "uuid": "GPU-test", "index": 0},
            "memory": {"total_mib": 6141},
        }
        with patch("calibration.gpu_context.GPUManager.calibration_status",
                   return_value=gpu):
            with self.assertRaisesRegex(RuntimeError, "Fonte"):
                search_gpu_context(
                    "qwen-2.5-7b-instruct", source="kaggle",
                    baseline_selection={
                        "source": "bartowski",
                        "native_verified_all_gpu": True,
                    },
                )

    def test_selected_kaggle_publisher_changes_profile_identity(self):
        base_handle = HANDLE
        other = "otherowner/qwen-2.5-7b-instruct/gguf/q5km/2"
        from calibration.profile_identity import resolve_identity
        gpu = {
            "ready": True, "errors": [], "foreign_processes": [],
            "gpu": {"vendor": "nvidia", "name": "Mock CUDA",
                    "uuid": "GPU-test", "index": 0},
            "memory": {"total_mib": 6141},
        }
        with (
            patch("calibration.profile_identity._ram_total_mib",
                  return_value=16000),
            patch("calibration.profile_identity._cpu_model",
                  return_value="Mock CPU"),
        ):
            a = resolve_identity(
                "qwen-2.5-7b-instruct",
                source="kaggle", kaggle_handle=base_handle, status=gpu,
            )
            b = resolve_identity(
                "qwen-2.5-7b-instruct",
                source="kaggle", kaggle_handle=other, status=gpu,
            )
        self.assertNotEqual(a["profile_key"], b["profile_key"])
        self.assertEqual(b["choices"]["repository"],
                         "otherowner/qwen-2.5-7b-instruct")
        self.assertEqual(a["choices"]["source"], "kaggle")

    def test_baseline_routes_kaggle_download_not_huggingface(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = "alternate/qwen-2.5-7b-instruct"
            handle = repo + "/gguf/mistral/2"
            local = Path(tmp) / "Qwen2.5-7B-Instruct-Q5_K_M.gguf"
            def fake_download(candidate, destination):
                self.assertEqual(candidate["kaggle_handle"], handle)
                self.assertEqual(Path(destination), Path(tmp))
                local.write_bytes(b"mock GGUF bytes")
                return local
            catalog = [{
                "filename": local.name, "quantization": "Q5_K_M",
                "quality_priority": 700, "kaggle_handle": handle,
                "repo_id": repo, "source_file_path": local.name,
            }]
            layout = {
                "metadata": {"general.architecture": "qwen2", "qwen2.block_count": 32},
                "weights_bytes": 5_000_000_000,
            }
            gpu = {
                "ready": True, "errors": [], "foreign_processes": [],
                "gpu": {"vendor": "nvidia", "name": "Mock NVIDIA",
                        "uuid": "GPU-mock", "index": 0},
                "memory": {"total_mib": 6141.0},
            }
            result_probe = {
                "success": True, "architecture": "qwen2",
                "full_gpu_native_verification": {"verified": True},
                "measurements": {"minimal_inference_completed": True},
            }
            with (
                patch("calibration.baseline_discovery.GPUManager.calibration_status",
                      return_value=gpu),
                patch("calibration.kaggle_source.inventory_kaggle",
                      return_value=catalog),
                patch("calibration.kaggle_source.download_kaggle",
                      side_effect=fake_download) as kaggle_download,
                patch("calibration.baseline_discovery.inspect_gguf",
                      return_value=layout),
                patch("calibration.baseline_discovery.probe_gpu_candidate",
                      return_value=result_probe),
                patch("huggingface_hub.hf_hub_download") as hf,
            ):
                result = choose_baseline(
                    root=Path(tmp), repo_id=repo,
                    model_id="qwen-2.5-7b-instruct", source="kaggle",
                    kaggle_handle=handle, persist_selection=False,
                )
            self.assertTrue(result["success"])
            self.assertEqual(result["baseline"]["source"], "kaggle")
            self.assertEqual(result["baseline"]["kaggle_handle"], handle)
            self.assertEqual(result["baseline"]["quantization"], "Q5_K_M")
            kaggle_download.assert_called_once()
            hf.assert_not_called()

    def test_source_identity_is_scoped_and_no_cross_source_fallback(self):
        from calibration.profile_identity import validate_choices
        for model in ("gemma-2-9b-it", "qwen-2.5-7b-instruct"):
            self.assertEqual(
                validate_choices(model, "kaggle")["source"], "kaggle"
            )
            self.assertEqual(
                validate_choices(model, "bartowski")["source"], "bartowski"
            )
        for model in ("mistral-7b-instruct-v0.3",
                      "llama-3.1-8b-instruct"):
            self.assertEqual(
                validate_choices(model, "bartowski")["source"], "bartowski"
            )
            with self.assertRaisesRegex(ValueError, "Fonte"):
                validate_choices(model, "kaggle")

    def test_official_kaggle_handles_are_defaults_only_for_two_models(self):
        from model import ModelManager
        self.assertEqual(
            ModelManager.source_spec("gemma-2-9b-it", "kaggle")["kaggle_handle"],
            "google/gemma-2/gguf/2.0-9b-it",
        )
        self.assertEqual(
            ModelManager.source_spec("qwen-2.5-7b-instruct", "kaggle")[
                "kaggle_handle"
            ], "qwen-lm/qwen2.5/gguf/7b-instruct",
        )
        for model in ("llama-3.1-8b-instruct",
                      "mistral-7b-instruct-v0.3"):
            self.assertEqual(ModelManager.available_sources(model),
                             ("bartowski",))

    def test_unrecognized_gguf_name_is_reported_but_not_ranked(self):
        unknown = subprocess.CompletedProcess(
            [], 0, "Name,Size\\nunknown-format.gguf,3GB\\n", ""
        )
        with patch("calibration.kaggle_source.subprocess.run",
                   return_value=unknown):
            found = inventory_kaggle(
                REPOSITORY, "Qwen2.5-7B-Instruct-", selected_handle=HANDLE
            )
        self.assertEqual(len(found), 1)
        self.assertIsNone(found[0]["quality_priority"])
        self.assertFalse(found[0]["filename_matches_model_prefix"])


if __name__ == "__main__":
    unittest.main()
