"""Phase 05 candidate discovery/quality/VRAM prefilter: offline, no weights."""
import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from ruth_runtime.candidate_selection import (
    POLICY_PATH, CandidateError, load_quantization_policy,
    inspect_single_nvidia_gpu, discover_quantizations, recommend_quantization,
)
from ruth_runtime.model_catalog import select_model
from ruth_runtime.launcher import main

REVISION = "d731033f3dc4018261fd39896e50984d398b4ac5"
UUID = "GPU-5a841828-882b-85d9-c98e-3a0d29e66e00"
PCI = "00000000:01:00.0"
Q4_SIZE = 5761057728  # metadata from the user's real HF lookup
SIZES = {
    "Q5_K_M": 6650000000,
    "Q4_K_M": Q4_SIZE,
    "Q4_K_S": 5480000000,
    "IQ4_XS": 5180000000,
    "Q3_K_L": 5130000000,
    "Q3_K_M": 4760000000,
    "Q2_K": 3810000000,
    "Q4_0_4_4": 5440000000,  # architecture-specific, always unranked
}


def metadata(variant_sizes=SIZES, *, wrong_pin=False, revision=REVISION,
             license_name="gemma", gated=False, missing_lfs=False):
    choice = select_model("production")
    return {
        "sha": revision, "gated": gated, "private": False,
        "cardData": {"license": license_name},
        "siblings": [
            {
                "rfilename": "gemma-2-9b-it-" + q + ".gguf",
                "size": size,
                **({} if missing_lfs else {
                    "lfs": {"sha256": ("0" * 64 if wrong_pin else
                                        choice.expected_sha256)
                            if q == "Q4_K_M" else "a" * 64,
                            "size": size},
                }),
            }
            for q, size in variant_sizes.items()
        ] + [{"rfilename": "README.md", "size": 200}],
    }


def llama_f16_metadata(*, f16_size=2480000000, include_f16=True):
    """Synthetic fixture: not a downloaded, pinned or verified F16 GGUF."""
    choice = select_model("test")
    files = [
        {
            "rfilename": choice.filename,
            "size": choice.expected_size_bytes,
            "lfs": {"sha256": choice.expected_sha256,
                    "size": choice.expected_size_bytes},
        },
    ]
    if include_f16:
        files.append({
            "rfilename": "Llama-3.2-1B-Instruct-f16.gguf",
            "size": f16_size,
            "lfs": {"sha256": "b" * 64, "size": f16_size},
        })
    return {
        "sha": choice.revision, "gated": False, "private": False,
        "cardData": {"license": choice.license},
        "siblings": files,
    }


def nvidia(*args, **kwargs):
    return SimpleNamespace(
        returncode=0,
        stdout=f"{UUID}, {PCI}, 6141, 5759\n",
    )


class QuantizationPolicyTests(unittest.TestCase):
    def test_policy_is_heuristic_not_measured_quality(self):
        obj = load_quantization_policy()
        self.assertEqual(obj["schema"], "ruth-quantization-policy-v2")
        self.assertNotIn("default_provisional_reserve_mib", obj)
        self.assertEqual(obj["initial_probe_n_ctx"], 128)
        self.assertIn("HEURISTIC", obj["description"])
        self.assertEqual(obj["excluded_on_cuda"]["Q4_0_4_4"],
                         "ARM-specific layout")

    def test_policy_rejects_duplicate_priority(self):
        data = load_quantization_policy()
        data["quality_policy"][1]["priority"] = data["quality_policy"][0]["priority"]
        with TemporaryDirectory() as temp:
            path = Path(temp) / POLICY_PATH
            path.parent.mkdir()
            path.write_text(json.dumps(data))
            with self.assertRaises(CandidateError):
                load_quantization_policy(Path(temp))

    def test_no_authorization_makes_no_network_or_gpu_calls(self):
        with patch("ruth_runtime.candidate_selection.fetch_huggingface_metadata",
                   side_effect=AssertionError("No HF request")), \
             patch("ruth_runtime.candidate_selection.inspect_single_nvidia_gpu",
                   side_effect=AssertionError("No GPU request")):
            result = recommend_quantization(authorize_metadata_network=False)
            listing = discover_quantizations(authorize_metadata_network=False)
        self.assertEqual(result["status"], "metadata_network_not_authorized")
        self.assertEqual(listing["status"], "metadata_network_not_authorized")
        self.assertFalse(result["download_started"])

    def test_exact_known_sha_validation_and_metadata_listing(self):
        result = discover_quantizations(
            authorize_metadata_network=True, fetch=lambda repo: metadata())
        self.assertEqual(result["status"], "quantizations_discovered", result)
        self.assertEqual(result["revision"], REVISION)
        variants = {x["quantization"]: x for x in result["quantizations"]}
        self.assertEqual(variants["Q4_K_M"]["size_bytes"], Q4_SIZE)
        self.assertTrue(variants["Q4_K_M"]["catalog_pin_exists"])
        self.assertFalse(variants["Q4_K_M"]["download_pin_ready"])
        self.assertFalse(variants["IQ4_XS"]["catalog_pin_exists"])
        self.assertFalse(variants["IQ4_XS"]["full_gpu_residency_verified"])
        self.assertEqual(variants["Q4_0_4_4"]["exclusion_reason"],
                         "ARM-specific layout")
        self.assertEqual(result["license"], "gemma")
        self.assertFalse(result["catalog_modified"])
        self.assertFalse(result["download_started"])

    def test_f16_is_discovered_on_llama_1b_not_skipped(self):
        result = discover_quantizations(
            purpose="test", authorize_metadata_network=True,
            fetch=lambda _: llama_f16_metadata(),
        )
        self.assertEqual(result["status"], "quantizations_discovered", result)
        files = {item["quantization"]: item for item in result["quantizations"]}
        self.assertIn("f16", files)
        self.assertEqual(files["f16"]["filename"],
                         "Llama-3.2-1B-Instruct-f16.gguf")
        self.assertEqual(files["f16"]["quality_priority_heuristic"], 220)
        self.assertIsNone(files["f16"]["exclusion_reason"])
        self.assertFalse(files["f16"]["download_pin_ready"])

    def test_llama_f16_passes_six_gb_prefilter_and_outranks_q4(self):
        result = recommend_quantization(
            purpose="test", authorize_metadata_network=True,
            fetch=lambda _: llama_f16_metadata(), run=nvidia,
        )
        self.assertEqual(result["status"], "provisional_shortlist_ready", result)
        self.assertEqual(result["first_candidate_quantization"], "f16")
        self.assertEqual(result["candidates"][0]["size_bytes"], 2480000000)
        self.assertFalse(result["candidate_is_production_qualified"])
        self.assertFalse(result["candidates"][0]["full_gpu_residency_verified"])
        self.assertFalse(result["catalog_modified"])
        self.assertFalse(result["download_started"])

    def test_f16_larger_than_free_vram_is_filtered_without_fake_fit(self):
        result = recommend_quantization(
            purpose="test", authorize_metadata_network=True,
            fetch=lambda _: llama_f16_metadata(f16_size=18000000000),
            run=nvidia,
        )
        self.assertEqual(result["first_candidate_quantization"], "Q4_K_M")
        f16 = next(x for x in result["all_quantizations"]
                   if x["quantization"] == "f16")
        self.assertEqual(f16["size_only_prefilter"],
                         "gguf_file_larger_than_free_vram")
        self.assertNotIn("f16", [c["quantization"] for c in result["candidates"]])

    def test_real_gemma_q4_kl_bytes_outrank_q4_km_without_reserve(self):
        # The two exact sizes come from the user's actual HF metadata output.
        real_bytes = {"Q4_K_L": 5983265728, "Q4_K_M": 5761057728}
        result = recommend_quantization(
            purpose="production", authorize_metadata_network=True,
            fetch=lambda _: metadata(real_bytes), run=nvidia,
        )
        self.assertEqual(result["first_candidate_quantization"], "Q4_K_L")
        self.assertEqual(result["candidates"][0]["size_bytes"], 5983265728)
        self.assertGreater(result["candidates"][0]["estimated_remaining_mib_after_file"], 0)

    def test_gemma_f16_absent_in_selected_repo_not_invented(self):
        result = discover_quantizations(
            purpose="production", authorize_metadata_network=True,
            fetch=lambda _: metadata(),
        )
        self.assertEqual(result["status"], "quantizations_discovered")
        self.assertNotIn("f16", [x["quantization"]
                                 for x in result["quantizations"]])

    def test_known_catalog_pin_mismatch_refused(self):
        result = discover_quantizations(
            authorize_metadata_network=True,
            fetch=lambda _: metadata(wrong_pin=True))
        self.assertEqual(result["status"], "quantization_discovery_blocked")
        self.assertIn("pin", result["error"])

    def test_missing_lfs_returns_blocked_not_guessed_sizes(self):
        result = discover_quantizations(
            authorize_metadata_network=True,
            fetch=lambda _: metadata(missing_lfs=True))
        self.assertEqual(result["status"], "quantization_discovery_blocked")

    def test_duplicate_filename_refused(self):
        obj = metadata()
        obj["siblings"].append(obj["siblings"][0])
        result = discover_quantizations(
            authorize_metadata_network=True, fetch=lambda _: obj)
        self.assertEqual(result["status"], "quantization_discovery_blocked")

    def test_reject_changed_license(self):
        result = discover_quantizations(
            authorize_metadata_network=True,
            fetch=lambda _: metadata(license_name="other"))
        self.assertEqual(result["status"], "quantization_discovery_blocked")

    def test_gated_model_not_processed(self):
        result = discover_quantizations(
            authorize_metadata_network=True, fetch=lambda _: metadata(gated="auto"))
        self.assertEqual(result["status"], "quantization_discovery_blocked")

    def test_main_revision_and_sha_are_reported_without_pinning(self):
        result = discover_quantizations(
            authorize_metadata_network=True, fetch=lambda _: metadata())
        self.assertEqual(result["revision"], REVISION)
        self.assertFalse(any(q["download_pin_ready"] for q in result["quantizations"]))

    def test_hardware_free_memory_is_read_correctly(self):
        data = inspect_single_nvidia_gpu(run=nvidia)
        self.assertEqual(data.free_mib, 5759)
        self.assertEqual(data.total_mib, 6141)
        self.assertEqual(data.uuid, UUID)

    def test_multi_gpu_or_invalid_memory_refused(self):
        def two_gpus(*args, **kwargs):
            one = nvidia().stdout
            return SimpleNamespace(returncode=0, stdout=one+one)
        with self.assertRaises(CandidateError):
            inspect_single_nvidia_gpu(run=two_gpus)
        with self.assertRaises(CandidateError):
            inspect_single_nvidia_gpu(
                run=lambda *a, **k: SimpleNamespace(
                    returncode=0, stdout=f"{UUID}, {PCI}, 6141, 7000\n"))

    def test_recommendation_uses_entire_free_vram_and_quality_heuristic(self):
        result = recommend_quantization(
            authorize_metadata_network=True,
            fetch=lambda _: metadata(), run=nvidia,
        )
        self.assertEqual(result["status"], "provisional_shortlist_ready", result)
        self.assertEqual(result["first_candidate_quantization"], "Q4_K_M")
        self.assertNotIn("reserve_mib", result)
        self.assertEqual(result["size_budget_mib"], 5759)
        self.assertEqual(result["size_budget_bytes"], 5759 * 1024 * 1024)
        self.assertEqual(result["initial_probe_n_ctx"], 128)
        self.assertTrue(result["runtime_memory_calibration_pending"])
        self.assertTrue(result["context_calibration_pending"])
        self.assertFalse(result["candidate_is_production_qualified"])
        self.assertFalse(result["catalog_modified"])
        self.assertFalse(result["gpu_inference_executed"])
        self.assertEqual(result["candidates"][0]["size_only_prefilter"],
                         "pass_requires_native_test")
        self.assertEqual(result["candidates"][0]["quantization"], "Q4_K_M")

    def test_lower_available_vram_selects_next_quality_variant(self):
        def lower_free(*args, **kwargs):
            return SimpleNamespace(
                returncode=0, stdout=f"{UUID}, {PCI}, 6141, 5000\n")
        result = recommend_quantization(
            authorize_metadata_network=True,
            fetch=lambda _: metadata(), run=lower_free)
        self.assertEqual(result["status"], "provisional_shortlist_ready")
        self.assertEqual(result["first_candidate_quantization"], "IQ4_XS")
        self.assertEqual(result["size_budget_mib"], 5000)
        self.assertNotIn("Q4_K_M", [c["quantization"] for c in result["candidates"]])

    def test_selection_does_not_claim_full_gpu_fit(self):
        result = recommend_quantization(
            authorize_metadata_network=True,
            fetch=lambda _: metadata(), run=nvidia)
        self.assertEqual(result["first_candidate_quantization"], "Q4_K_M")
        self.assertFalse(result["candidate_is_production_qualified"])
        self.assertFalse(result["candidates"][0]["full_gpu_residency_verified"])
        self.assertEqual(result["selection_rule"],
                         "gguf_size_bytes_le_current_free_vram_bytes")

    def test_exact_file_size_equal_to_free_vram_is_eligible(self):
        free_bytes = 5759 * 1024 * 1024
        result = recommend_quantization(
            authorize_metadata_network=True,
            fetch=lambda _: metadata({"Q3_K_M": free_bytes}),
            run=nvidia)
        self.assertEqual(result["first_candidate_quantization"], "Q3_K_M")
        self.assertEqual(result["candidates"][0]["estimated_remaining_mib_after_file"], 0)
        self.assertFalse(result["candidate_is_production_qualified"])

    def test_file_one_byte_larger_than_free_vram_is_not_eligible(self):
        free_bytes = 5759 * 1024 * 1024
        result = recommend_quantization(
            authorize_metadata_network=True,
            fetch=lambda _: metadata({"Q3_K_M": free_bytes + 1}),
            run=nvidia)
        self.assertEqual(result["status"], "no_quantization_with_gguf_file_within_free_vram")
        self.assertEqual(result["candidates"], [])
        self.assertFalse(result["runtime_profile_published"])

    def test_insufficient_free_vram_has_no_magic_fallback(self):
        def busy_gpu(*args, **kwargs):
            return SimpleNamespace(
                returncode=0, stdout=f"{UUID}, {PCI}, 6141, 0\n")
        result = recommend_quantization(
            authorize_metadata_network=True,
            fetch=lambda _: metadata(), run=busy_gpu)
        self.assertEqual(result["status"], "no_quantization_with_gguf_file_within_free_vram")
        self.assertIsNone(result["first_candidate_quantization"])
        self.assertEqual(result["candidates"], [])
        self.assertFalse(result["runtime_profile_published"])

    def test_invalid_limit_blocks_before_download(self):
        for limit in (0, 11, -1, True):
            with self.subTest(limit=limit):
                result = recommend_quantization(
                    authorize_metadata_network=True,
                    max_candidates=limit,
                    fetch=lambda _: self.fail("No HF fetch"),
                    run=lambda *a, **k: self.fail("No GPU fetch"))
                self.assertEqual(result["status"], "provisional_selection_blocked")

    def test_legacy_reserve_policy_is_rejected(self):
        data = load_quantization_policy()
        data["default_provisional_reserve_mib"] = 768
        with TemporaryDirectory() as temp:
            path = Path(temp) / POLICY_PATH
            path.parent.mkdir()
            path.write_text(json.dumps(data))
            with self.assertRaises(CandidateError):
                load_quantization_policy(Path(temp))


class ModelChoiceCliTests(unittest.TestCase):
    def invoke(self, opts):
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            ret = main(["models", *opts])
        return ret, json.loads(stdout.getvalue())

    def test_quantization_cli_needs_metadata_flag(self):
        with patch("ruth_runtime.candidate_selection.fetch_huggingface_metadata",
                   side_effect=AssertionError("No network")):
            code, result = self.invoke(["quantizations", "--purpose", "production"])
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "metadata_network_not_authorized")

    def test_recommend_cli_needs_metadata_flag(self):
        code, result = self.invoke(["recommend", "--purpose", "production"])
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "metadata_network_not_authorized")

    def test_cli_listing_is_metadata_only(self):
        with patch("ruth_runtime.candidate_selection.fetch_huggingface_metadata",
                   return_value=metadata()):
            code, result = self.invoke(
                ["quantizations", "--purpose", "production",
                 "--allow-metadata-network"])
        self.assertEqual(code, 0, result)
        self.assertEqual(result["status"], "quantizations_discovered")
        self.assertFalse(result["download_started"])

    def test_removed_reserve_argument_is_not_accepted_by_cli(self):
        with patch("sys.stderr", new_callable=StringIO):
            with self.assertRaises(SystemExit) as err:
                main(["models", "recommend", "--purpose", "production",
                      "--reserve-mib", "768", "--allow-metadata-network"])
        self.assertEqual(err.exception.code, 2)

    def test_cli_candidate_recommendation_is_read_only(self):
        with patch("ruth_runtime.candidate_selection.fetch_huggingface_metadata",
                   return_value=metadata()), \
             patch("ruth_runtime.candidate_selection.inspect_single_nvidia_gpu",
                   return_value=SimpleNamespace(
                       uuid=UUID, pci_bus=PCI, free_mib=5759, total_mib=6141,
                       to_dict=lambda: {"uuid": UUID, "free_mib": 5759,
                                        "total_mib": 6141, "pci_bus": PCI})):
            code, result = self.invoke([
                "recommend", "--purpose", "production",
                "--allow-metadata-network", "--top-k", "2",
            ])
        self.assertEqual(code, 0, result)
        self.assertEqual(result["first_candidate_quantization"], "Q4_K_M")
        self.assertEqual(result["initial_probe_n_ctx"], 128)
        self.assertEqual(len(result["candidates"]), 2)
        self.assertFalse(result["gpu_inference_executed"])
        self.assertFalse(result["download_started"])


if __name__ == "__main__":
    unittest.main()
