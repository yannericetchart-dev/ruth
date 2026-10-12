"""Data-driven architectural lower-bound tests; no real network or GPU."""

import unittest
from unittest.mock import patch

from calibration.kaggle_architecture_bound import (
    mandatory_dense_parameter_floor, conservative_kaggle_weight_floor,
)

MODEL = "qwen-2.5-7b-instruct"
REPO = "qwen-lm/qwen2.5"
HANDLE = REPO + "/gguf/7b-instruct/1"
VRAM = 6141 * 1048576
CONFIG = {
    "model_type": "qwen2",
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
}
REF = {"repo": "Qwen/Qwen2.5-7B-Instruct", "config": CONFIG}


def item(quant: str, repo: str = REPO, handle: str = HANDLE) -> dict:
    return {
        "quantization": quant, "repo_id": repo,
        "kaggle_handle": handle,
        "filename": f"Qwen2.5-7B-Instruct-{quant}.gguf",
    }


class ArchitectureBoundTests(unittest.TestCase):
    def test_calculated_geometry_not_model_specific_numbers(self):
        self.assertEqual(
            mandatory_dense_parameter_floor(CONFIG), 7_070_285_824
        )
        second = {
            **CONFIG, "model_type": "llama",
            "num_hidden_layers": 12, "hidden_size": 512,
            "intermediate_size": 2048, "num_attention_heads": 8,
            "num_key_value_heads": 2, "vocab_size": 32000,
        }
        manual = (
            32000 * 512
            + 12 * (3 * 512 * 2048 + 2 * 512 * 512
                    + 2 * 512 * 2 * 64)
        )
        self.assertEqual(mandatory_dense_parameter_floor(second), manual)
        self.assertNotEqual(manual, mandatory_dense_parameter_floor(CONFIG))

    def test_fp16_and_q8_excluded_from_calculated_config(self):
        fp16 = conservative_kaggle_weight_floor(
            MODEL, item("fp16"), VRAM, reference=REF
        )
        q8 = conservative_kaggle_weight_floor(
            MODEL, item("q8_0"), VRAM, reference=REF
        )
        self.assertTrue(fp16["exceeds_total_vram"])
        self.assertTrue(q8["exceeds_total_vram"])
        self.assertEqual(fp16["minimum_dense_parameters"], 7_070_285_824)
        self.assertEqual(fp16["minimum_weight_bytes"], 14_140_571_648)
        self.assertEqual(q8["minimum_weight_bytes"], 7_070_285_824)

    def test_missing_reference_never_invents_exclusion(self):
        self.assertIsNone(
            conservative_kaggle_weight_floor(
                MODEL, item("fp16"), VRAM
            )
        )

    def test_scoped_to_registered_kaggle_source_and_handle(self):
        for model, candidate in (
            (MODEL, item("fp16", repo="other/qwen")),
            (MODEL, item("fp16", handle="other/qwen/gguf/7b-instruct")),
            (MODEL, item("fp16", handle=HANDLE + "/unexpected")),
            ("llama-3.1-8b-instruct", item("fp16")),
        ):
            with self.subTest(candidate=candidate, model=model):
                self.assertIsNone(
                    conservative_kaggle_weight_floor(
                        model, candidate, VRAM, reference=REF
                    )
                )

    def test_generalizes_to_another_registered_kaggle_model(self):
        second_config = {
            **CONFIG, "model_type": "llama",
            "num_hidden_layers": 12, "hidden_size": 512,
            "intermediate_size": 2048, "num_attention_heads": 8,
            "num_key_value_heads": 2, "vocab_size": 32000,
        }
        spec = {
            "source": "kaggle", "repository": "org/model",
            "kaggle_handle": "org/model/gguf/standard",
            "gguf_prefix": "Another-",
            "architecture": "llama",
            "architecture_reference_repo": "org/model-original",
        }
        another = {
            "repo_id": "org/model",
            "kaggle_handle": "org/model/gguf/standard/3",
            "filename": "Another-fp16.gguf",
            "quantization": "fp16",
        }
        with patch(
            "calibration.kaggle_architecture_bound.ModelManager.source_spec",
            return_value=spec,
        ):
            outcome = conservative_kaggle_weight_floor(
                "another-7b-instruct", another, VRAM,
                reference={"repo": "org/model-original", "config": second_config},
            )
        self.assertIsNotNone(outcome)
        self.assertEqual(
            outcome["minimum_weight_bytes"],
            mandatory_dense_parameter_floor(second_config) * 2,
        )
        self.assertFalse(outcome["exceeds_total_vram"])

    def test_unknown_architectures_or_moe_cannot_be_excluded(self):
        for config in (
            {**CONFIG, "model_type": "unknown"},
            {**CONFIG, "num_local_experts": 8},
            {**CONFIG, "intermediate_size": None},
            {**CONFIG, "num_attention_heads": 0},
        ):
            with self.subTest(config=config):
                self.assertIsNone(mandatory_dense_parameter_floor(config))
                self.assertIsNone(
                    conservative_kaggle_weight_floor(
                        MODEL, item("fp16"), VRAM,
                        reference={"repo": REF["repo"], "config": config},
                    )
                )

    def test_other_quantizations_not_improperly_excluded(self):
        for quant in ("q6_k", "q5_k_m", "q5_0", "q4_k_m", "q3_k_m"):
            with self.subTest(quant=quant):
                self.assertIsNone(
                    conservative_kaggle_weight_floor(
                        MODEL, item(quant), VRAM, reference=REF
                    )
                )

    def test_larger_vram_does_not_exclude_q8(self):
        outcome = conservative_kaggle_weight_floor(
            MODEL, item("q8_0"), 12 * 1024**3, reference=REF
        )
        self.assertFalse(outcome["exceeds_total_vram"])


if __name__ == "__main__":
    unittest.main()
