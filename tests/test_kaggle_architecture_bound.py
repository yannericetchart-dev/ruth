"""Qwen 2.5 7B architecture-only, full-GPU feasibility exclusions."""

import unittest

from calibration.kaggle_architecture_bound import (
    conservative_kaggle_weight_floor,
)

MODEL = "qwen-2.5-7b-instruct"
REPO = "qwen-lm/qwen2.5"
HANDLE = "qwen-lm/qwen2.5/gguf/7b-instruct/1"
VRAM = 6141 * 1048576


def item(quant: str, repo: str = REPO, handle: str = HANDLE) -> dict:
    return {
        "quantization": quant,
        "repo_id": repo,
        "kaggle_handle": handle,
        "filename": f"Qwen2.5-7B-Instruct-{quant}.gguf",
    }


class QwenArchitectureBoundTests(unittest.TestCase):
    def test_fp16_and_q8_rejected_above_rtx_4050_total_vram(self):
        fp16 = conservative_kaggle_weight_floor(
            MODEL, item("fp16"), VRAM
        )
        q8 = conservative_kaggle_weight_floor(
            MODEL, item("q8_0"), VRAM
        )
        self.assertTrue(fp16["exceeds_total_vram"])
        self.assertTrue(q8["exceeds_total_vram"])
        self.assertEqual(
            fp16["minimum_dense_parameters"], 7_070_285_824
        )
        self.assertEqual(
            fp16["minimum_weight_bytes"], 14_140_571_648
        )
        self.assertEqual(
            q8["minimum_weight_bytes"], 7_070_285_824
        )
        self.assertGreater(q8["minimum_weight_mib"], 6141)

    def test_source_scoping_prevents_fallback_or_foreign_exclusion(self):
        for model, candidate in (
            (MODEL, item("fp16", repo="other/qwen")),
            (MODEL, item("fp16", handle="different/qwen/gguf/7b-instruct")),
            (MODEL, item("fp16", handle=HANDLE + "/unexpected")),
            ("llama-3.1-8b-instruct", item("fp16")),
        ):
            with self.subTest(model=model, candidate=candidate):
                self.assertIsNone(
                    conservative_kaggle_weight_floor(model, candidate, VRAM)
                )

    def test_near_vram_quants_are_not_discarded_without_real_proof(self):
        for quant in ("q6_k", "q5_k_m", "q5_0", "q4_k_m", "q3_k_m"):
            with self.subTest(quant=quant):
                self.assertIsNone(
                    conservative_kaggle_weight_floor(
                        MODEL, item(quant), VRAM
                    )
                )

    def test_larger_vram_does_not_exclude_q8_without_capacity_reason(self):
        value = conservative_kaggle_weight_floor(
            MODEL, item("Q8_0"), 12 * 1024**3
        )
        self.assertFalse(value["exceeds_total_vram"])

    def test_unversioned_canonical_variant_is_supported(self):
        value = conservative_kaggle_weight_floor(
            MODEL, item("fp16", handle=HANDLE.rsplit("/", 1)[0]), VRAM
        )
        self.assertTrue(value["exceeds_total_vram"])


if __name__ == "__main__":
    unittest.main()
