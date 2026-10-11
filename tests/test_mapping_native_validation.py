"""CPU-only unit tests for native mapping selection and log verification."""

import unittest

from calibration.mapping_candidates import propose_mapping_candidates
from calibration.mapping_native_validation import (
    choose_native_trials, verify_override_logs,
)


def fixture():
    tensors = [{"name": "token_embd.weight", "group": "embedding", "block": None, "bytes": 100}]
    for block in range(4):
        tensors.extend([
            {"name": f"blk.{block}.attn_k.weight", "block": block, "group": "attention", "bytes": 20},
            {"name": f"blk.{block}.attn_q.weight", "block": block, "group": "attention", "bytes": 30},
            {"name": f"blk.{block}.ffn_down.weight", "block": block, "group": "feed_forward", "bytes": 40},
        ])
    return {
        "metadata": {"general.architecture": "gemma2", "gemma2.block_count": 4},
        "tensors": tensors,
    }


def log_for(candidate, override=None, output=True):
    lines = []
    if output:
        lines.append("llama_model_load_tensors: offloading output layer to GPU")
    for name in candidate["host_tensor_names"]:
        buft = (override or {}).get(name, "CUDA_Host")
        lines.append(
            f"llama_model_loader: tensor {name} (20 MiB Q4) "
            f"buffer type overridden to {buft}"
        )
    for name in candidate["gpu_tensor_names"]:
        buft = (override or {}).get(name, "CUDA0")
        lines.append(
            f"llama_model_loader: tensor {name} (20 MiB Q4) "
            f"buffer type overridden to {buft}"
        )
    return "\n".join(lines)


class NativeMappingValidationTests(unittest.TestCase):
    def setUp(self):
        self.report = propose_mapping_candidates(fixture(), 3)
        self.selection = choose_native_trials(self.report)

    def test_three_distinct_directions(self):
        self.assertEqual(
            [x["direction"] for x in self.selection],
            ["gpu_to_host", "host_to_gpu", "swap"],
        )
        self.assertTrue(all(x["family"] == "attention" for x in self.selection))

    def test_every_target_verified(self):
        for candidate in self.selection:
            evidence = verify_override_logs(candidate, log_for(candidate))
            self.assertTrue(evidence["verified"])
            self.assertTrue(evidence["native_logged_names_exact"])
            self.assertTrue(evidence["output_gpu_verified_by_log"])

    def test_cuda_host_is_host_not_gpu(self):
        candidate = self.selection[1]  # GPU promotion
        name = candidate["gpu_tensor_names"][0]
        evidence = verify_override_logs(
            candidate, log_for(candidate, {name: "CUDA_Host"})
        )
        self.assertFalse(evidence["verified"])
        self.assertEqual(evidence["tensors"][0]["native_domain"], "host_cuda_pinned")

    def test_missing_output_log_fails(self):
        candidate = self.selection[0]
        evidence = verify_override_logs(candidate, log_for(candidate, output=False))
        self.assertFalse(evidence["verified"])

    def test_missing_or_extra_native_log_fails(self):
        candidate = self.selection[2]
        evidence = verify_override_logs(candidate, log_for(candidate) + (
            "\nllama_model_loader: tensor blk.99.attn_q.weight "
            "(2 MiB Q4) buffer type overridden to CUDA0"
        ))
        self.assertFalse(evidence["verified"])
        self.assertFalse(evidence["native_logged_names_exact"])

        evidence = verify_override_logs(
            candidate, log_for(candidate).splitlines()[0]
        )
        self.assertFalse(evidence["verified"])


if __name__ == "__main__":
    unittest.main()
