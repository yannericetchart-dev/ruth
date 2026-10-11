"""Tensor-family classification tests for all non-repeating GGUF weights."""

import unittest

from model.inspect import _kind


class TensorClassificationTests(unittest.TestCase):
    def test_input_and_output(self):
        self.assertEqual(_kind("token_embd.weight"), ("embedding", None))
        self.assertEqual(_kind("output.weight"), ("output", None))
        self.assertEqual(_kind("output_norm.weight"), ("output", None))

    def test_repeating_tensors_without_weight_suffix(self):
        self.assertEqual(_kind("blk.11.ffn_down.weight"), ("feed_forward", 11))
        self.assertEqual(_kind("blk.11.attn_k.weight"), ("attention", 11))
        self.assertEqual(_kind("blk.11.ffn_norm.weight"), ("normalization", 11))
        self.assertEqual(_kind("blk.11.custom.scale"), ("other_block", 11))


if __name__ == "__main__":
    unittest.main()
