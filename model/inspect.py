"""Analisi in sola lettura di struttura e dimensioni dei tensori GGUF."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _kind(name: str) -> tuple[str, int | None]:
    if name.startswith("token_embd."):
        return "embedding", None
    if name.startswith(("output.", "output_norm.")):
        return "output", None
    match = re.match(r"^blk\.(\d+)\.(.+)$", name)
    if not match:
        return "other", None
    block = int(match.group(1))
    part = match.group(2)
    if "norm" in part:
        return "normalization", block
    if part.startswith("attn_"):
        return "attention", block
    if part.startswith("ffn_"):
        return "feed_forward", block
    return "other_block", block


def _field(reader, name: str):
    field = reader.fields.get(name)
    if field is None or field.types[0].name == "ARRAY":
        return None
    return field.contents()


_SPLIT_SUFFIX = re.compile(
    r"^(?P<stem>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)


def gguf_file_set(path: str | Path) -> list[Path]:
    """Return a complete, sequential GGUF shard set, or fail closed.

    llama.cpp requires the first split and automatically looks for siblings.
    Do not infer a valid model from one isolated shard.
    """
    path = Path(path).expanduser().resolve()
    match = _SPLIT_SUFFIX.fullmatch(path.name)
    if match is None:
        return [path]
    part, count = int(match["part"]), int(match["total"])
    if not 1 <= part <= count <= 99999:
        raise ValueError("GGUF: indice shard non valido.")
    if part != 1:
        raise ValueError("GGUF: caricare soltanto dal primo shard.")
    names = [
        path.with_name(
            f"{match['stem']}-{i:05d}-of-{count:05d}.gguf"
        )
        for i in range(1, count + 1)
    ]
    missing = [p.name for p in names if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"GGUF: gruppo shard incompleto: {len(missing)} file mancanti."
        )
    return names


def inspect_gguf(path: str | Path, n_ctx: int = 4096) -> dict:
    """Legge il catalogo tensori senza eseguire inferenza o allocare VRAM."""
    from gguf import GGUFReader

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GGUF non trovato: {path}")
    if n_ctx < 1:
        raise ValueError("n_ctx deve essere positivo")

    files = gguf_file_set(path)
    readers = [GGUFReader(file, mode="r") for file in files]
    reader = readers[0]
    architecture = _field(reader, "general.architecture")
    for index, shard in enumerate(readers):
        shard_arch = _field(shard, "general.architecture")
        if shard_arch is not None and shard_arch != architecture:
            raise ValueError("GGUF: architetture shard incompatibili.")
        split_total = _field(shard, "split.count")
        split_index = _field(shard, "split.no")
        if split_total is not None and int(split_total) != len(files):
            raise ValueError("GGUF: conteggio shard incompatibile.")
        if split_index is not None and int(split_index) != index:
            raise ValueError("GGUF: ordine shard incompatibile.")
    prefix = f"{architecture}." if architecture else ""
    keys = [
        "general.architecture", "general.name", "general.file_type",
        "general.quantization_version", "general.alignment",
    ] + [prefix + key for key in (
        "block_count", "context_length", "embedding_length",
        "feed_forward_length", "attention.head_count", "attention.head_count_kv",
        "attention.key_length", "attention.value_length", "attention.sliding_window",
    )]
    metadata = {key: value for key in keys if (value := _field(reader, key)) is not None}

    tensors = []
    groups = {}
    layers = {}
    types = Counter()
    total_weights = 0
    for shard_index, tensor in (
        (index, tensor)
        for index, shard in enumerate(readers)
        for tensor in shard.tensors
    ):
        kind, layer_id = _kind(tensor.name)
        size = int(tensor.n_bytes)
        record = {
            "name": tensor.name,
            "shape": [int(dim) for dim in tensor.shape],
            "type": tensor.tensor_type.name,
            "elements": int(tensor.n_elements),
            "bytes": size,
            "offset": int(tensor.data_offset),
            "offset_source": "GGUFReader.data_offset",
            "shard_index": shard_index,
            "shard_file": files[shard_index].name,
            "group": kind,
            "block": layer_id,
        }
        tensors.append(record)
        total_weights += size
        types[record["type"]] += 1
        group = groups.setdefault(kind, {"count": 0, "bytes": 0})
        group["count"] += 1
        group["bytes"] += size
        if layer_id is not None:
            block = layers.setdefault(str(layer_id), {"count": 0, "bytes": 0, "groups": {}})
            block["count"] += 1
            block["bytes"] += size
            block["groups"][kind] = block["groups"].get(kind, 0) + size

    if len(tensors) != len({tensor["name"] for tensor in tensors}):
        raise ValueError("GGUF: nomi tensori duplicati tra shard.")

    # Stima teorica con K/V f16 complete; non rappresenta l'allocazione
    # reale di llama.cpp (SWA, quantizzazione KV e buffer dipendono dal runtime).
    kv = {
        "n_ctx": n_ctx,
        "assumption": "all blocks, full n_ctx, K/V f16",
        "estimated_bytes": None,
        "note": "Solo geometria teorica: non indica VRAM usata né buffer CUDA.",
    }
    n_blocks = metadata.get(prefix + "block_count")
    kv_heads = metadata.get(prefix + "attention.head_count_kv", metadata.get(prefix + "attention.head_count"))
    embed = metadata.get(prefix + "embedding_length")
    head_count = metadata.get(prefix + "attention.head_count")
    key_dim = metadata.get(prefix + "attention.key_length")
    value_dim = metadata.get(prefix + "attention.value_length")
    if key_dim is None and embed is not None and head_count:
        key_dim = int(embed) // int(head_count)
    if value_dim is None:
        value_dim = key_dim
    if all(isinstance(v, int) and v > 0 for v in (n_blocks, kv_heads, key_dim, value_dim)):
        kv["estimated_bytes"] = int(n_blocks) * n_ctx * int(kv_heads) * (int(key_dim) + int(value_dim)) * 2

    return {
        "model_path": str(path),
        "file_bytes": sum(file.stat().st_size for file in files),
        "shard_files": [str(file) for file in files],
        "shard_count": len(files),
        "metadata": metadata,
        "tensor_count": len(tensors),
        "weights_bytes": total_weights,
        "quantization_counts": dict(sorted(types.items())),
        "groups": dict(sorted(groups.items())),
        "layers": dict(sorted(layers.items(), key=lambda kv: int(kv[0]))),
        "kv_cache_theoretical": kv,
        "runtime_memory_note": (
            "GGUF non specifica picchi CUDA, scratch buffer o mapping reale. "
            "Saranno misurati in una successiva fase di calibrazione."
        ),
        "tensors": tensors,
    }


def _mib(n: int) -> str:
    return f"{n / (1024 * 1024):,.1f} MiB"


def main() -> None:
    from .manager import ModelManager

    parser = argparse.ArgumentParser(description="Analisi GGUF in sola lettura (senza GPU)")
    parser.add_argument("--model", default="gemma-2-9b-it", help="ID ModelManager")
    parser.add_argument("--ctx", type=int, default=4096, help="Contesto per la stima KV")
    parser.add_argument("--json", action="store_true", help="JSON con ogni tensore")
    parser.add_argument("--layers", action="store_true", help="Dimensioni per blocco transformer")
    args = parser.parse_args()

    path = ModelManager().get(args.model)
    if path is None:
        parser.error("GGUF non installato; scaricalo prima con ModelManager.download()")
    report = inspect_gguf(path, n_ctx=args.ctx)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print(f"GGUF: {path.name}")
    print(f"Dimensione file: {_mib(report['file_bytes'])}")
    print(f"Pesi tensori: {_mib(report['weights_bytes'])} in {report['tensor_count']} tensori")
    print("\nMetadati del modello:")
    for key, value in report["metadata"].items():
        print(f"  {key}: {value}")
    print("\nGruppi di tensori (byte GGUF, non VRAM):")
    for group, info in report["groups"].items():
        print(f"  {group:16} {info['count']:4} tensori   {_mib(info['bytes'])}")
    print("\nTipi di quantizzazione:")
    for quant, count in report["quantization_counts"].items():
        print(f"  {quant}: {count}")
    if args.layers:
        print("\nPesi per blocco transformer:")
        for index, info in report["layers"].items():
            print(f"  blk.{index:>2}: {info['count']} tensori, {_mib(info['bytes'])}")
    kv = report["kv_cache_theoretical"]
    print(f"\nKV teorica F16 per n_ctx={args.ctx}: " + (
        _mib(kv["estimated_bytes"]) if kv["estimated_bytes"] is not None else "non stimabile dai metadati"
    ))
    print("Nota: " + kv["note"])
    print(report["runtime_memory_note"])
    print("Per l'elenco dei tensori: --json (puoi reindirizzare l'output fuori dal repo).")


if __name__ == "__main__":
    main()
