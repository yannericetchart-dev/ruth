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
    def test_cli_failures_are_classified_without_leaking_credentials(self):
        from calibration.kaggle_source import (
            _csv_call, classify_kaggle_cli_failure,
        )

        examples = (
            ("Kaggle credentials not found", "credentials_missing"),
            ("API token missing", "credentials_missing"),
            ("Authentication required", "credentials_missing"),
            ("User has not consented to terms of use 403", "license_required"),
            ("Consent required for this model", "license_required"),
            ("You must accept the license", "license_required"),
            ("403 resource requiring consent", "forbidden"),
            ("HTTPError 401", "unauthorized"),
            ("HTTP 403 Forbidden", "forbidden"),
            ("404 Not Found", "not_found"),
            ("Model variation not found", "not_found"),
            ("File not found", "unknown"),
            ("429 Too Many Requests", "rate_limited"),
            ("500 Internal Server Error", "server_error"),
            ("usage: kaggle models variations", "cli_syntax"),
            ("ConnectionError", "network"),
            ("unknown failure", "unknown"),
        )
        for diagnostic, category in examples:
            with self.subTest(diagnostic=diagnostic):
                observed, _ = classify_kaggle_cli_failure(
                    diagnostic + " KGAT_test_private"
                )
                self.assertEqual(observed, category)

        with patch("calibration.kaggle_source.subprocess.run",
                   return_value=subprocess.CompletedProcess(
                       [], 1, "",
                       "HTTP 403 Forbidden token=KGAT_sensitive"
                   )):
            with self.assertRaisesRegex(RuntimeError, "categoria=forbidden") as e:
                _csv_call("variations", "files", HANDLE)
        self.assertNotIn("KGAT_sensitive", str(e.exception))

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

    def test_latest_404_recovers_only_from_listed_numeric_version(self):
        unversioned = REPOSITORY + "/gguf/7b-instruct"
        calls = []

        def simulate(argv, **kwargs):
            calls.append(argv)
            if argv[2:4] == ["variations", "files"]:
                return subprocess.CompletedProcess(argv, 1, "", "404 Not Found")
            if argv[2:5] == ["variations", "versions", "list"]:
                return subprocess.CompletedProcess(
                    argv, 0, "version,variation\n2,7b-instruct\n8,7b-instruct\n", ""
                )
            if argv[2:5] == ["variations", "versions", "files"]:
                return subprocess.CompletedProcess(
                    argv, 0,
                    "Name,Size\nQwen2.5-7B-Instruct-Q4_K_M.gguf,4.2GB\n", ""
                )
            raise AssertionError("Unexpected CLI command")

        with patch("calibration.kaggle_source.subprocess.run",
                   side_effect=simulate):
            models = inventory_kaggle(
                REPOSITORY, "Qwen2.5-7B-Instruct-",
                selected_handle=unversioned,
            )
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["kaggle_handle"], unversioned + "/8")
        self.assertEqual([x[2:5] for x in calls], [
            ["variations", "files", unversioned],
            ["variations", "versions", "list"],
            ["variations", "versions", "files"],
        ])
        self.assertEqual(calls[-1][5], unversioned + "/8")

    def test_latest_404_never_invents_version_if_discovery_fails(self):
        unversioned = REPOSITORY + "/gguf/7b-instruct"
        with patch("calibration.kaggle_source.subprocess.run",
                   return_value=subprocess.CompletedProcess(
                       [], 1, "", "404 Not Found"
                   )) as cli:
            with self.assertRaisesRegex(RuntimeError, "categoria=not_found"):
                inventory_kaggle(
                    REPOSITORY, "Qwen2.5-7B-Instruct-",
                    selected_handle=unversioned,
                )
        self.assertEqual(cli.call_count, 2)

    def test_version_file_failure_retains_secondary_safe_cause(self):
        handle = REPOSITORY + "/gguf/7b-instruct"
        calls = []
        def simulate(argv, **kwargs):
            calls.append(argv)
            if argv[2:4] == ["variations", "files"]:
                return subprocess.CompletedProcess(argv, 1, "", "404 Not Found")
            if argv[2:5] == ["variations", "versions", "list"]:
                return subprocess.CompletedProcess(
                    argv, 0, "versionNumber\n5\n", ""
                )
            if argv[2:5] == ["variations", "versions", "files"]:
                return subprocess.CompletedProcess(
                    argv, 1, "", "403 Forbidden token=KGAT_PRIVATE"
                )
            raise AssertionError("Unexpected CLI command")
        with patch("calibration.kaggle_source.subprocess.run",
                   side_effect=simulate):
            with self.assertRaises(RuntimeError) as caught:
                inventory_kaggle(
                    REPOSITORY, "Qwen2.5-7B-Instruct-",
                    selected_handle=handle,
                )
        message = str(caught.exception)
        self.assertIn("categoria=not_found", message)
        self.assertIn("Versione 5 confermata", message)
        self.assertIn("categoria=forbidden", message)
        self.assertNotIn("KGAT_PRIVATE", message)
        self.assertEqual(len(calls), 3)

    def test_forbidden_variation_never_retries_other_endpoints(self):
        unversioned = REPOSITORY + "/gguf/7b-instruct"
        with patch("calibration.kaggle_source.subprocess.run",
                   return_value=subprocess.CompletedProcess(
                       [], 1, "", "403 Forbidden"
                   )) as cli:
            with self.assertRaisesRegex(RuntimeError, "categoria=forbidden"):
                inventory_kaggle(
                    REPOSITORY, "Qwen2.5-7B-Instruct-",
                    selected_handle=unversioned,
                )
        cli.assert_called_once()

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
            "../../gguf/quantizations",
            "other/../gguf/quantizations",
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

    def test_sharded_gguf_inventory_groups_whole_quantizations(self):
        from calibration.kaggle_source import inventory_kaggle
        files = [
            {"Name": f"qwen2.5-7b-instruct-q5_k_m-{i:05d}-of-00002.gguf"}
            for i in (1, 2)
        ] + [
            {"Name": f"qwen2.5-7b-instruct-fp16-{i:05d}-of-00004.gguf"}
            for i in (1, 2, 3, 4)
        ] + [{"Name": "Qwen2.5-7B-Instruct-Q4_K_M.gguf"}]
        with patch("calibration.kaggle_source.variation_files",
                   return_value=(files, HANDLE)):
            found = inventory_kaggle(
                REPOSITORY, "Qwen2.5-7B-Instruct-",
                selected_handle=HANDLE,
            )
        self.assertEqual(len(found), 3)
        by_quant = {record["quantization"].casefold(): record for record in found}
        self.assertEqual(set(by_quant), {"fp16", "q5_k_m", "q4_k_m"})
        fp16 = by_quant["fp16"]
        self.assertEqual(fp16["shard_count"], 4)
        self.assertEqual(len(fp16["shard_filenames"]), 4)
        self.assertTrue(fp16["split_complete"])
        self.assertEqual(fp16["shard_filenames"][0],
                         "qwen2.5-7b-instruct-fp16-00001-of-00004.gguf")
        self.assertTrue(all(
            record["quality_priority"] is not None for record in found
        ))
        self.assertLess(fp16["quality_priority"],
                        by_quant["q5_k_m"]["quality_priority"])

    def test_multi_shard_kaggle_sizes_are_summed_without_download(self):
        from calibration.kaggle_source import parse_kaggle_file_size
        self.assertEqual(
            parse_kaggle_file_size("2.5GB")["bytes"], 2_500_000_000
        )
        self.assertEqual(
            parse_kaggle_file_size("1 GiB")["bytes"], 1_073_741_824
        )
        self.assertEqual(
            parse_kaggle_file_size("5,000")["bytes"], 5000
        )
        self.assertIsNone(parse_kaggle_file_size("unknown"))

        files = [
            {"Name": "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
             "Size": "2.5GB"},
            {"Name": "qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf",
             "Size": "2.0GB"},
            {"Name": "qwen2.5-7b-instruct-q4_k_m.gguf",
             "Size": "unknown"},
        ]
        with (patch("calibration.kaggle_source.variation_files",
                    return_value=(files, HANDLE)),
              patch("kagglehub.model_download") as download):
            found = inventory_kaggle(
                REPOSITORY, "Qwen2.5-7B-Instruct-",
                selected_handle=HANDLE
            )
        download.assert_not_called()
        self.assertEqual(len(found), 2)
        by_quant = {r["quantization"].casefold(): r for r in found}
        self.assertEqual(
            by_quant["q5_k_m"]["reported_download_bytes"], 4_500_000_000
        )
        self.assertTrue(
            by_quant["q5_k_m"]["reported_download_size_estimated"]
        )
        self.assertIsNone(
            by_quant["q4_k_m"]["reported_download_bytes"]
        )
        self.assertEqual(
            len(by_quant["q5_k_m"]["reported_shard_sizes"]), 2
        )

    def test_incomplete_split_remains_unranked_without_download(self):
        files = [
            {"Name": "qwen2.5-7b-instruct-q5_k_m-00002-of-00003.gguf"},
            {"Name": "qwen2.5-7b-instruct-q5_k_m-00003-of-00003.gguf"},
        ]
        with patch("calibration.kaggle_source.variation_files",
                   return_value=(files, HANDLE)):
            found = inventory_kaggle(
                REPOSITORY, "Qwen2.5-7B-Instruct-",
                selected_handle=HANDLE,
            )
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]["split_complete"])
        self.assertEqual(found[0]["missing_shard_indices"], [1])
        self.assertIsNone(found[0]["quality_priority"])
        self.assertFalse(found[0]["cuda_baseline_eligible"])
        with patch("kagglehub.model_download") as download:
            with self.assertRaisesRegex(ValueError, "shard"):
                download_kaggle(found[0], Path("/not-used"))
            download.assert_not_called()

    def test_split_inventory_rejects_conflicting_totals_and_duplicates(self):
        cases = (
            [{"Name": "qwen-q4-00001-of-00002.gguf"},
             {"Name": "qwen-q4-00002-of-00003.gguf"}],
            [{"Name": "qwen-q4-00001-of-00002.gguf"},
             {"Name": "qwen-q4-00001-of-00002.gguf"}],
        )
        for rows in cases:
            with self.subTest(rows=rows):
                with patch("calibration.kaggle_source.variation_files",
                           return_value=(rows, HANDLE)):
                    with self.assertRaisesRegex(RuntimeError, "shard"):
                        inventory_kaggle(
                            REPOSITORY, "Qwen2.5-7B-Instruct-",
                            selected_handle=HANDLE,
                        )

    def test_download_multiple_shards_before_returning_first(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "cache"
            source.mkdir()
            names = [
                f"qwen2.5-7b-instruct-q4_k_m-{i:05d}-of-00002.gguf"
                for i in (1, 2)
            ]
            for name in names:
                (source / name).write_bytes(b"fake-GGUF")
            candidate = {
                "filename": names[0], "source_file_path": names[0],
                "repo_id": REPOSITORY, "kaggle_handle": HANDLE,
                "shard_filenames": names,
                "shard_source_file_paths": names,
                "shard_count": 2, "split_complete": True,
            }
            def fetch(handle, path):
                self.assertEqual(handle, HANDLE)
                return str(source / path)
            with patch("kagglehub.model_download",
                       side_effect=fetch) as download:
                first = download_kaggle(candidate, root / "models")
            self.assertEqual(first.name, names[0])
            self.assertEqual(download.call_count, 2)
            self.assertTrue(all((root / "models" / n).is_file()
                                for n in names))
            with patch("kagglehub.model_download") as cached:
                self.assertEqual(
                    download_kaggle(candidate, root / "models"), first
                )
                cached.assert_not_called()

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

    def test_only_qwen_kaggle_variation_is_used_for_inventory(self):
        from model import ModelManager
        model = "qwen-2.5-7b-instruct"
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

    def test_gemma_kaggle_is_not_a_supported_source(self):
        from model import ModelManager
        from calibration.profile_identity import validate_choices
        self.assertEqual(
            ModelManager.available_sources("gemma-2-9b-it"),
            ("bartowski",)
        )
        with self.assertRaisesRegex(ValueError, "Fonte"):
            ModelManager.source_spec("gemma-2-9b-it", "kaggle")
        with self.assertRaisesRegex(ValueError, "Fonte"):
            validate_choices("gemma-2-9b-it", "kaggle")
        with self.assertRaisesRegex(ValueError, "Fonte"):
            choose_baseline(
                model_id="gemma-2-9b-it", source="kaggle",
                inventory_only=True,
                kaggle_handle="google/gemma-2/gguf/2.0-9b-it/1",
            )

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

    def test_context_allows_version_pinned_by_kaggle_catalog(self):
        from calibration.gpu_context import search_gpu_context
        source_handle = "qwen-lm/qwen2.5/gguf/7b-instruct"
        pinned_handle = source_handle + "/8"
        gpu = {
            "ready": True, "errors": [], "foreign_processes": [],
            "gpu": {"vendor": "nvidia", "name": "Mock CUDA",
                    "uuid": "GPU-test", "index": 0},
            "memory": {"total_mib": 6141},
        }
        with patch("calibration.gpu_context.GPUManager.calibration_status",
                   return_value=gpu):
            selection = {
                "source": "kaggle", "kaggle_handle": pinned_handle,
                "repository": REPOSITORY, "model_id": "qwen-2.5-7b-instruct",
                "native_verified_all_gpu": True,
                "local_path": "/GGUF-NOT-DOWNLOADED.gguf",
                "filename": "GGUF-NOT-DOWNLOADED.gguf",
            }
            # Passing the handle-match checks proves that /8 was accepted
            # as a pinned snapshot of the unversioned catalogue variation.
            with self.assertRaisesRegex(RuntimeError, "GGUF selezionato assente"):
                search_gpu_context(
                    "qwen-2.5-7b-instruct", source="kaggle",
                    kaggle_handle=source_handle,
                    baseline_selection=selection,
                )
            with self.assertRaisesRegex(RuntimeError, "Variante/versione"):
                search_gpu_context(
                    "qwen-2.5-7b-instruct", source="kaggle",
                    kaggle_handle=source_handle + "/7",
                    baseline_selection=selection,
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
                "reported_shard_sizes": [{"bytes": 1024, "estimated": False}],
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
                    kaggle_max_download_mib=1, kaggle_min_free_mib=0,
                )
            self.assertTrue(result["success"])
            self.assertEqual(result["baseline"]["source"], "kaggle")
            self.assertEqual(result["baseline"]["kaggle_handle"], handle)
            self.assertEqual(result["baseline"]["quantization"], "Q5_K_M")
            kaggle_download.assert_called_once()
            hf.assert_not_called()

    def test_source_identity_is_scoped_and_no_cross_source_fallback(self):
        from calibration.profile_identity import validate_choices
        self.assertEqual(
            validate_choices("qwen-2.5-7b-instruct", "kaggle")["source"],
            "kaggle",
        )
        for model in (
            "gemma-2-9b-it", "qwen-2.5-7b-instruct",
            "llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3",
        ):
            self.assertEqual(
                validate_choices(model, "bartowski")["source"],
                "bartowski",
            )
        for model in (
            "gemma-2-9b-it", "llama-3.1-8b-instruct",
            "mistral-7b-instruct-v0.3",
        ):
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, "Fonte"):
                    validate_choices(model, "kaggle")

    def test_only_qwen_has_kaggle_handle_in_catalog(self):
        from model import ModelManager
        self.assertEqual(
            ModelManager.source_spec("qwen-2.5-7b-instruct", "kaggle")[
                "kaggle_handle"
            ], "qwen-lm/qwen2.5/gguf/7b-instruct",
        )
        for model in (
            "gemma-2-9b-it", "llama-3.1-8b-instruct",
            "mistral-7b-instruct-v0.3",
        ):
            self.assertEqual(
                ModelManager.available_sources(model), ("bartowski",)
            )

    def test_unrecognized_gguf_name_is_reported_but_not_ranked(self):
        unknown = subprocess.CompletedProcess(
            [], 0, "Name,Size\nunknown-format.gguf,3GB\n", ""
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
