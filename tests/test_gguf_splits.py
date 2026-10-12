"""CPU-only tests for complete GGUF split metadata and tensor verification."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from model.inspect import gguf_file_set, inspect_gguf
from calibration.baseline_discovery import all_gpu_rules


class GGUFShardInspectionTests(unittest.TestCase):
    def test_first_shard_resolves_complete_set(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            shards = [
                folder / f"qwen-quant-{i:05d}-of-00002.gguf"
                for i in (1, 2)
            ]
            for p in shards:
                p.write_bytes(b"metadata-only-test")
            self.assertEqual(gguf_file_set(shards[0]), shards)
            with self.assertRaisesRegex(ValueError, "primo shard"):
                gguf_file_set(shards[1])
            shards[1].unlink()
            with self.assertRaisesRegex(FileNotFoundError, "incompleto"):
                gguf_file_set(shards[0])

    def test_inspect_collects_tensor_metadata_across_all_shards(self):
        def field(value):
            return SimpleNamespace(
                types=[SimpleNamespace(name="UINT32")],
                contents=lambda: value,
            )

        def tensor(name, size):
            return SimpleNamespace(
                name=name, n_bytes=size, shape=[2],
                tensor_type=SimpleNamespace(name="Q4_K"),
                n_elements=2, data_offset=0,
            )

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            paths = [
                directory / f"qwen-quant-{index:05d}-of-00002.gguf"
                for index in (1, 2)
            ]
            for path in paths:
                path.write_bytes(b"some bytes")

            reader_data = {
                str(paths[0]): SimpleNamespace(
                    fields={
                        "general.architecture": field("qwen2"),
                        "qwen2.block_count": field(28),
                        "qwen2.context_length": field(32768),
                        "split.no": field(0),
                        "split.count": field(2),
                    },
                    tensors=[
                        tensor("token_embd.weight", 100),
                        tensor("blk.0.attn_q.weight", 200),
                    ],
                ),
                str(paths[1]): SimpleNamespace(
                    fields={
                        "general.architecture": field("qwen2"),
                        "split.no": field(1),
                        "split.count": field(2),
                    },
                    tensors=[
                        tensor("output.weight", 300),
                        tensor("blk.0.ffn_up.weight", 400),
                    ],
                ),
            }
            with patch("gguf.GGUFReader",
                       side_effect=lambda path, mode: reader_data[str(path)]) as readers:
                result = inspect_gguf(paths[0], n_ctx=64)
            self.assertEqual(readers.call_count, 2)
            self.assertEqual(result["shard_count"], 2)
            self.assertEqual(result["tensor_count"], 4)
            self.assertEqual(result["weights_bytes"], 1000)
            self.assertEqual(
                result["file_bytes"], sum(p.stat().st_size for p in paths)
            )
            self.assertEqual(result["metadata"]["qwen2.block_count"], 28)
            rules, expected = all_gpu_rules(result)
            self.assertTrue(rules)
            self.assertEqual(len(expected["gpu_tensor_names"]), 4)

    def test_split_metadata_mismatch_is_not_accepted(self):
        def field(value):
            return SimpleNamespace(
                types=[SimpleNamespace(name="UINT32")],
                contents=lambda: value,
            )
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            first = directory / "qwen-quant-00001-of-00002.gguf"
            second = directory / "qwen-quant-00002-of-00002.gguf"
            first.write_bytes(b"x")
            second.write_bytes(b"x")
            mock_first = SimpleNamespace(
                fields={"general.architecture": field("qwen2"),
                        "split.no": field(0), "split.count": field(2)},
                tensors=[],
            )
            mock_second = SimpleNamespace(
                fields={"general.architecture": field("qwen2"),
                        "split.no": field(8), "split.count": field(2)},
                tensors=[],
            )
            with patch("gguf.GGUFReader",
                       side_effect=[mock_first, mock_second]):
                with self.assertRaisesRegex(ValueError, "ordine shard"):
                    inspect_gguf(first, n_ctx=64)


if __name__ == "__main__":
    unittest.main()
