"""Model catalog / Hugging Face metadata regression suite (no downloads)."""
import json
from pathlib import Path
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

from ruth_runtime.dependencies import PROJECT_ROOT
from ruth_runtime.model_catalog import (
    CATALOG_FILE, CatalogError, load_catalog, list_models,
    select_model, resolve_model_metadata,
)
from ruth_runtime.launcher import main


def metadata(choice, *, revision=None, sha=None, filename=None,
             size=808000000, license_name=None, gated=False):
    return {
        "sha": revision if revision is not None else (choice.revision or "1" * 40), "private": False, "gated": gated,
        "cardData": {"license": license_name or choice.license},
        "siblings": [
            {"rfilename": filename or choice.filename, "size": size,
             "lfs": {"sha256": sha or choice.expected_sha256, "size": size}}
        ],
    }


class CatalogTests(unittest.TestCase):
    def test_four_families_and_kaggle_todo(self):
        catalog = list_models()
        self.assertEqual(catalog["default_source"], "huggingface")
        self.assertEqual([x["family"] for x in catalog["families"]],
                         ["gemma", "llama", "qwen", "mistral"])
        self.assertEqual(catalog["sources"]["kaggle"]["status"], "TODO")
        self.assertEqual(catalog["families"][2]["status"],
                         "registered_no_verified_candidate")
        self.assertEqual(catalog["families"][3]["status"],
                         "registered_no_verified_candidate")
        self.assertFalse(catalog["download_started"])

    def test_default_production_is_gemma_2_9b_it(self):
        selected = select_model("production")
        self.assertEqual(selected.path, "gemma/gemma-2/gemma-2-9B-it/gguf/Q4_K_M")
        self.assertEqual(selected.repo_id, "bartowski/gemma-2-9b-it-GGUF")
        self.assertEqual(selected.license, "gemma")
        self.assertEqual(selected.assessment, "requires_real_hardware_profiling")
        self.assertEqual(len(selected.expected_sha256), 64)
        self.assertFalse(selected.downloadable)
        self.assertFalse(selected.local_model_ready)
        self.assertIsNone(selected.revision)

    def test_default_lightweight_test_is_llama_3_2_1b(self):
        selected = select_model("test")
        self.assertEqual(selected.path, "llama/llama-3.2/llama-3.2-1B-instruct/gguf/Q4_K_M")
        self.assertEqual(selected.repo_id, "bartowski/Llama-3.2-1B-Instruct-GGUF")
        self.assertEqual(selected.license, "llama3.2")
        self.assertEqual(selected.assessment, "lightweight_functional_tests_only")
        self.assertEqual(selected.expected_sha256,
                         "6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83")
        self.assertFalse(selected.gpu_offload_executed)
        self.assertEqual(selected.revision, "067b946cf014b7c697f3654f621d577a3e3afd1c")
        self.assertEqual(selected.expected_size_bytes, 807694464)

    def test_explicit_hierarchical_selection(self):
        selected = select_model(
            "test", family="gemma", model="gemma-2",
            version="gemma-2-9B-it", format="gguf", quantization="Q4_K_M",
        )
        self.assertEqual(selected.family, "gemma")
        self.assertEqual(selected.purpose, "test")

    def test_unregistered_model_fails_closed_without_fallback(self):
        with self.assertRaises(CatalogError):
            select_model("test", family="qwen")
        with self.assertRaises(CatalogError):
            select_model("test", quantization="F16")
        with self.assertRaises(CatalogError):
            select_model("test", format="safetensors")

    def test_invalid_purpose_refused(self):
        with self.assertRaises(CatalogError):
            select_model("benchmarks")

    def test_catalog_rejects_invalid_checksum_and_duplicate(self):
        data = load_catalog()
        data["models"][0]["variants"][0]["expected_sha256"] = "0bad"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / CATALOG_FILE
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(data))
            with self.assertRaises(CatalogError):
                load_catalog(Path(d))

    def test_catalog_rejects_kaggle_as_implemented(self):
        data = load_catalog()
        data["sources"]["kaggle"]["status"] = "implemented"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / CATALOG_FILE
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(data))
            with self.assertRaises(CatalogError):
                load_catalog(Path(d))


class HfMetadataTests(unittest.TestCase):
    def test_no_network_authorization_does_not_call_any_fetch(self):
        selected = select_model()
        def never(_):
            self.fail("No HF network access without permission")
        result = resolve_model_metadata(selected, fetch=never)
        self.assertEqual(result["status"], "metadata_network_not_authorized")
        self.assertFalse(result["download_started"])
        self.assertFalse(result["license_accepted"])

    def test_metadata_verification_pins_commit_and_lfs_sha(self):
        selected = select_model()
        observed = []
        def fake(repo):
            observed.append(repo)
            return metadata(selected)
        response = resolve_model_metadata(
            selected, authorize_metadata_network=True, fetch=fake)
        self.assertEqual(observed, [selected.repo_id])
        self.assertEqual(response["status"], "source_metadata_verified_license_pending")
        self.assertEqual(response["revision"], selected.revision)
        self.assertEqual(response["sha256"], selected.expected_sha256)
        self.assertEqual(response["size_bytes"], 808000000)
        self.assertIn("/blob/" + selected.revision + "/", response["model_url"])
        self.assertFalse(response["license_accepted"])
        self.assertFalse(response["download_started"])
        self.assertFalse(response["gpu_offload_executed"])

    def test_sha_mismatch_refused(self):
        selected = select_model()
        response = resolve_model_metadata(
            selected, authorize_metadata_network=True,
            fetch=lambda _: metadata(selected, sha="0" * 64))
        self.assertEqual(response["status"], "source_metadata_unverified")
        self.assertNotIn("model_url", response)

    def test_unpinned_revision_refused(self):
        selected = select_model()
        result = resolve_model_metadata(
            selected, authorize_metadata_network=True,
            fetch=lambda _: metadata(selected, revision="main"))
        self.assertEqual(result["status"], "source_metadata_unverified")

    def test_remote_byte_size_changed_is_refused(self):
        selected = select_model()
        response = resolve_model_metadata(
            selected, authorize_metadata_network=True,
            fetch=lambda _: metadata(selected, size=123456789))
        self.assertEqual(response["status"], "source_metadata_unverified")

    def test_remote_revision_changed_is_refused(self):
        selected = select_model()
        response = resolve_model_metadata(
            selected, authorize_metadata_network=True,
            fetch=lambda _: metadata(selected, revision="f" * 40))
        self.assertEqual(response["status"], "source_metadata_unverified")

    def test_missing_file_refused(self):
        selected = select_model()
        result = resolve_model_metadata(
            selected, authorize_metadata_network=True,
            fetch=lambda _: metadata(selected, filename="other.gguf"))
        self.assertEqual(result["status"], "source_metadata_unverified")

    def test_no_lfs_metadata_refused(self):
        selected = select_model()
        result = resolve_model_metadata(
            selected, authorize_metadata_network=True,
            fetch=lambda _: {"sha": "a" * 40, "siblings": [
                {"rfilename": selected.filename, "size": 808000000}
            ]})
        self.assertEqual(result["status"], "source_metadata_unverified")

    def test_license_change_refused(self):
        selected = select_model()
        result = resolve_model_metadata(
            selected, authorize_metadata_network=True,
            fetch=lambda _: metadata(selected, license_name="other"))
        self.assertEqual(result["status"], "source_metadata_unverified")

    def test_zero_or_missing_size_refused(self):
        selected = select_model()
        result = resolve_model_metadata(
            selected, authorize_metadata_network=True,
            fetch=lambda _: metadata(selected, size=0))
        self.assertEqual(result["status"], "source_metadata_unverified")

    def test_source_gated_is_reported_not_accepted(self):
        selected = select_model("production")
        result = resolve_model_metadata(
            selected, authorize_metadata_network=True,
            fetch=lambda _: metadata(selected, gated="auto"))
        self.assertEqual(result["status"], "source_metadata_verified_license_pending")
        self.assertEqual(result["gated"], "auto")
        self.assertFalse(result["license_accepted"])


class ModelsCliTests(unittest.TestCase):
    def run_cli(self, args):
        with patch("sys.stdout", new_callable=StringIO) as output, \
             patch("ruth_runtime.launcher.inspect_python_environment",
                   side_effect=AssertionError("models must not inspect CUDA or venv")):
            result = main(["models", *args])
            return result, json.loads(output.getvalue())

    def test_models_list_is_read_only_without_probing_venv(self):
        code, result = self.run_cli(["list"])
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "catalog_ready")

    def test_models_select_production(self):
        code, result = self.run_cli(["select", "--purpose", "production"])
        self.assertEqual(code, 0)
        self.assertEqual(result["selection"]["version"], "gemma-2-9B-it")
        self.assertFalse(result["download_started"])

    def test_models_resolve_without_authorized_network_is_blocked(self):
        with patch("ruth_runtime.model_catalog.fetch_huggingface_metadata",
                   side_effect=AssertionError("No HF fetch allowed")):
            code, result = self.run_cli(["resolve", "--purpose", "test"])
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "metadata_network_not_authorized")

    def test_models_explicit_selection_does_not_auto_switch_family(self):
        code, result = self.run_cli(
            ["select", "--family", "mistral"])
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "catalog_selection_blocked")

    def test_models_resolve_metadata_only(self):
        selected = select_model("test")
        with patch("ruth_runtime.model_catalog.fetch_huggingface_metadata",
                   return_value=metadata(selected)):
            code, result = self.run_cli(
                ["resolve", "--purpose", "test", "--allow-metadata-network"])
        self.assertEqual(code, 0)
        self.assertFalse(result["download_started"])
        self.assertFalse(result["license_accepted"])
        self.assertEqual(result["status"], "source_metadata_verified_license_pending")


if __name__ == "__main__":
    unittest.main()
