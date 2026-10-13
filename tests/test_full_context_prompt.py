"""CPU-only regression tests for complete-context Phase 1 prompts."""

import unittest
from calibration.gpu_context import make_context_tokens


class FakeLLM:
    def tokenize(self, payload, add_bos=True, special=True):
        # Deterministic tokenizer stand-in; no CUDA, GGUF, or download needed.
        return list(range(len(payload.decode("utf-8").split()) + int(add_bos)))


class FullContextPromptTests(unittest.TestCase):
    def test_prompt_plus_generation_exactly_matches_context(self):
        for n_ctx, generated in ((64, 8), (256, 8), (512, 16), (1024, 32)):
            with self.subTest(n_ctx=n_ctx, generated=generated):
                prompt = make_context_tokens(
                    FakeLLM(), "gemma2", n_ctx, generated,
                    model_id="gemma-2-9b-it",
                )
                self.assertEqual(len(prompt), n_ctx - generated)
                self.assertEqual(len(prompt) + generated, n_ctx)

    def test_insufficient_room_is_rejected(self):
        with self.assertRaises(ValueError):
            make_context_tokens(FakeLLM(), "gemma2", 64, 60)


if __name__ == "__main__":
    unittest.main()
