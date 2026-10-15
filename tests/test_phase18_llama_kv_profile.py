"""Source-derived llama.cpp KV calculations. No live model or CUDA required."""
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ruth_runtime.dependencies import PROJECT_ROOT
from ruth_runtime.llama_kv_profile import calculate_llama_kv, pad


class LlamaKVProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "catalog").mkdir()
        shutil.copyfile(PROJECT_ROOT / "catalog/llama_kv_backend_policy.json",
                        self.root / "catalog/llama_kv_backend_policy.json")

    def calc(self, **kw):
        settings = dict(n_ctx=128, n_batch=64, n_ubatch=64,
                        swa_full=True, flash_attention="off",
                        project_root=self.root)
        settings.update(kw)
        return calculate_llama_kv(**settings)

    def test_context_pad_independent_of_flash_attention(self):
        for fa in ("off", "on", "auto"):
            for requested, effective in (
                (64, 256), (128, 256), (256, 256),
                (257, 512), (513, 768), (1024, 1024),
            ):
                with self.subTest(fa=fa, requested=requested):
                    self.assertEqual(
                        self.calc(n_ctx=requested, flash_attention=fa)
                        ["configuration"]["n_ctx_effective_derived"], effective)

    def test_f16_kv_for_gemma_2_9b_matches_historical_84mib(self):
        r = self.calc()
        self.assertEqual(r["total_logical_mib_derived"], 84.0)
        self.assertEqual(r["global_cache"]["layers"], 21)
        self.assertEqual(r["swa_cache"]["layers"], 21)
        self.assertEqual(r["key_logical_bytes_derived"], 42 * 1048576)
        self.assertEqual(r["value_logical_bytes_derived"], 42 * 1048576)
        self.assertIsNone(r["native_kv_buffer_mib_observed"])

    def test_swa_off_breakpoint_is_padded_window_plus_microbatch(self):
        for ctx, saved in ((4096, 0), (4352, 0), (4608, 42),
                           (5120, 126), (8192, 630)):
            with self.subTest(ctx=ctx):
                r = self.calc(n_ctx=ctx, swa_full=False)
                self.assertEqual(r["swa_full_off_saves_mib_derived"], saved)
                self.assertEqual(r["swa_off_first_saving_at_effective_ctx"], 4608)
                self.assertEqual(r["swa_cache"]["cells"], min(ctx, 4352))

    def test_ubatch_changes_swa_limit_only_at_padding_boundary(self):
        a = self.calc(n_ctx=4608, n_batch=256, n_ubatch=64, swa_full=False)
        b = self.calc(n_ctx=4608, n_batch=256, n_ubatch=128, swa_full=False)
        c = self.calc(n_ctx=4608, n_batch=256, n_ubatch=256, swa_full=False)
        self.assertEqual(a["swa_cache"]["cells"], 4352)
        self.assertEqual(b["swa_cache"]["cells"], 4352)
        self.assertEqual(c["swa_cache"]["cells"], 4352)
        d = self.calc(n_ctx=4608, n_batch=512, n_ubatch=512, swa_full=False)
        self.assertEqual(d["swa_cache"]["cells"], 4608)
        self.assertEqual(d["swa_full_off_saves_mib_derived"], 0)

    def test_logical_batch_does_not_raise_microbatch(self):
        result = self.calc(n_ctx=128, n_batch=1024, n_ubatch=64)
        self.assertEqual(result["configuration"]["n_batch_effective_derived"], 128)
        self.assertEqual(result["configuration"]["n_ubatch_effective_derived"], 64)

    def test_separate_quantized_and_mixed_key_value(self):
        for k,v,key_mib,value_mib in [
            ("F16", "F16", 42, 42),
            ("Q8_0", "F16", 22.3125, 42),
            ("Q8_0", "Q8_0", 22.3125, 22.3125),
            ("Q4_0", "Q8_0", 11.8125, 22.3125),
        ]:
            with self.subTest(k=k,v=v):
                x = self.calc(type_k=k, type_v=v, flash_attention="on")
                self.assertEqual(x["key_logical_bytes_derived"]/1048576,key_mib)
                self.assertEqual(x["value_logical_bytes_derived"]/1048576,value_mib)

    def test_quantized_v_refused_with_flash_off(self):
        with self.assertRaisesRegex(ValueError, "Quantized V requires"):
            self.calc(type_k="F16", type_v="Q8_0")
        self.assertEqual(self.calc(type_k="Q8_0", type_v="F16")
                         ["status"], "llama_kv_source_derived")

    def test_multisequence_and_patched_backend_not_assumed(self):
        with self.assertRaises(ValueError):
            self.calc(n_seq_max=2)
        with self.assertRaises(ValueError):
            self.calc(kv_unified=True)
        with self.assertRaises(ValueError):
            self.calc(type_v="UNKNOWN")
        policy = self.root / "catalog/llama_kv_backend_policy.json"
        import json
        payload = json.loads(policy.read_text())
        payload["context_alignment_cells"] = 128
        policy.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "Unverified"):
            self.calc()

    def test_padding_input_validation(self):
        self.assertEqual(pad(257,256),512)
        for v in (0, -1, 0.5, True):
            with self.assertRaises(ValueError):
                pad(v,256)
