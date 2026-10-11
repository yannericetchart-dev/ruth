"""Offline tests: native mapping composition requires non-overlapping targets."""

import unittest

from calibration.mapping_candidates import propose_mapping_candidates
from calibration.mapping_composition import compose_candidates
from calibration.mapping_native_validation import verify_override_logs


class CompositionTests(unittest.TestCase):
    @staticmethod
    def proposals():
        rows = [
            {"name": "token_embd.weight", "group": "embedding", "block": None, "bytes": 100},
            {"name": "output_norm.weight", "group": "output", "block": None, "bytes": 8},
        ]
        for block in range(4):
            rows.extend([
                {"name": f"blk.{block}.attn_k.weight", "group": "attention", "block": block, "bytes": 8},
                {"name": f"blk.{block}.ffn_gate.weight", "group": "feed_forward", "block": block, "bytes": 24},
                {"name": f"blk.{block}.ffn_up.weight", "group": "feed_forward", "block": block, "bytes": 24},
            ])
        layout = {
            "metadata": {"general.architecture": "gemma2", "gemma2.block_count": 4},
            "tensors": rows,
        }
        return propose_mapping_candidates(layout, 3)

    def test_composition_all_directions(self):
        proposals = self.proposals()
        combined = compose_candidates(proposals, [
            "single:gpu_to_host:blk.2.attn_k.weight",
            "single:host_to_gpu:blk.0.ffn_gate.weight",
        ])
        self.assertEqual(len(combined["tensor_names"]), 2)
        self.assertEqual(combined["estimated_vram_weight_delta_bytes"], 16)
        logs = "\n".join([
            "llama_model_load_tensors: offloading output layer to GPU",
            "llama_model_loader: tensor blk.2.attn_k.weight (1 MiB Q4) buffer type overridden to CUDA_Host",
            "llama_model_loader: tensor blk.0.ffn_gate.weight (1 MiB Q4) buffer type overridden to CUDA0",
        ])
        self.assertTrue(verify_override_logs(combined, logs)["verified"])

    def test_duplicate_and_overlapping_rules_rejected(self):
        proposals = self.proposals()
        with self.assertRaises(ValueError):
            compose_candidates(proposals, ["baseline"])
        with self.assertRaises(ValueError):
            compose_candidates(proposals, [])
        a = "single:gpu_to_host:blk.2.ffn_up.weight"
        with self.assertRaises(ValueError):
            compose_candidates(proposals, [a, a])
        with self.assertRaises(ValueError):
            compose_candidates(proposals, [
                a, "family:gpu_to_host:blk.2:feed_forward"
            ])

    def test_tied_output_can_compose_but_stays_coupled(self):
        proposals = self.proposals()
        comp = compose_candidates(proposals, [
            "tied:cpu:token_embd.weight",
            "single:host_to_gpu:blk.0.attn_k.weight",
        ])
        self.assertIsNone(comp["estimated_vram_weight_delta_bytes"])
        self.assertEqual(comp["expected_native_log_occurrences"], {"token_embd.weight": 2})
        logs = "\n".join([
            "llama_model_loader: tensor token_embd.weight (1 MiB Q4) buffer type overridden to CUDA_Host",
            "llama_model_loader: tensor token_embd.weight (1 MiB Q4) buffer type overridden to CUDA_Host",
            "llama_model_loader: tensor blk.0.attn_k.weight (1 MiB Q4) buffer type overridden to CUDA0",
        ])
        self.assertTrue(verify_override_logs(comp, logs)["verified"])

    def test_unrecognized_id_rejected(self):
        with self.assertRaises(ValueError):
            compose_candidates(self.proposals(), ["nonexistent"])
        

if __name__ == "__main__":
    unittest.main()
