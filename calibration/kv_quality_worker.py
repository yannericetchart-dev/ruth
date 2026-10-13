"""Isolated, memory-bounded teacher-forced KV quality worker.

The comparison deliberately holds model weights, FA setting and KV LOCATION
constant (CPU) for the f16 reference and quantized candidate. Latency of the
actual GPU-resident candidate is measured separately by gpu_context.probe_context.

Scores are sampled at deterministic next-token positions (not whole-corpus PPL).
We access the last-token logits directly from the 0.3.36 native context to avoid
allocating llama_cpp.Llama(logits_all=True)'s n_ctx x vocab score matrix.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from model.inspect import inspect_gguf
from .baseline_discovery import all_gpu_rules, native_confirmation
from .gpu_context import context_limit, validate_kv_config, verify_native_kv_config
from .mapping_phase import verify_mapping_kv_runtime
from .native_memory import parse_native_memory


def score_positions(n_ctx: int, count: int) -> list[int]:
    if n_ctx < 64 or count < 2:
        raise ValueError("Richiesti n_ctx >= 64 e almeno due punti qualità.")
    # Exclude first 16 tokens; last evaluated token must have a next token.
    return sorted(set(int(i) for i in np.linspace(
        16, n_ctx - 2, min(count, n_ctx - 18)
    )))


def logsumexp(logits: np.ndarray) -> float:
    z = np.asarray(logits, dtype=np.float64)
    top = float(z.max())
    if not math.isfinite(top):
        raise ValueError("Logits non finiti.")
    return top + math.log(float(np.exp(z - top).sum()))


def kl_nats(reference: np.ndarray, candidate: np.ndarray) -> float:
    p = np.asarray(reference, dtype=np.float64)
    q = np.asarray(candidate, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1 or not p.size:
        raise ValueError("Distribuzioni logits incompatibili.")
    zp, zq = logsumexp(p), logsumexp(q)
    logp, logq = p - zp, q - zq
    divergence = float(np.dot(np.exp(logp), logp - logq))
    if not math.isfinite(divergence):
        raise ValueError("KL non finita.")
    return max(0.0, divergence)  # float round-off can yield tiny negatives


def sampled_metrics(reference: np.ndarray, candidate: np.ndarray,
                    targets: list[int]) -> dict:
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != len(targets):
        raise ValueError("Logits e posizioni non corrispondono.")
    nll_a, nll_b, kls = [], [], []
    for j, target in enumerate(targets):
        if not 0 <= target < a.shape[1]:
            raise ValueError("Token target fuori vocabolario.")
        r, c = a[j], b[j]
        nll_a.append(logsumexp(r) - float(r[target]))
        nll_b.append(logsumexp(c) - float(c[target]))
        kls.append(kl_nats(r, c))
    return {
        "samples": len(targets),
        "reference_sampled_ppl": round(math.exp(sum(nll_a) / len(nll_a)), 6),
        "candidate_sampled_ppl": round(math.exp(sum(nll_b) / len(nll_b)), 6),
        "delta_mean_nll_nats": round(
            (sum(nll_b) - sum(nll_a)) / len(nll_a), 8
        ),
        "mean_kl_nats": round(sum(kls) / len(kls), 8),
        "max_kl_nats": round(max(kls), 8),
        "note": "PPL e KL campionate, teacher forcing, log-softmax sull'intero vocabolario.",
    }


def native_sample_logits(llm, ids: list[int], positions: list[int],
                         batch: int) -> np.ndarray:
    """Stream only last-token logit vectors, keeping KV prefix unchanged."""
    vocabulary = int(llm.n_vocab())
    outputs = np.empty((len(positions), vocabulary), dtype=np.float32)
    llm.reset()
    past = 0
    for index, position in enumerate(positions):
        end = position + 1
        while past < end:
            current = ids[past:min(end, past + batch)]
            llm._batch.set_batch(current, n_past=past, logits_all=False)
            llm._ctx.decode(llm._batch)
            past += len(current)
        ptr = llm._ctx.get_logits()
        if not ptr:
            raise RuntimeError("Puntatore logits nativo NULL.")
        vector = np.ctypeslib.as_array(ptr, shape=(vocabulary,))
        outputs[index, :] = vector
    if not np.isfinite(outputs).all():
        raise ValueError("Logits contengono NaN o inf.")
    return outputs


def quality_worker(*, model_path: Path, corpus: Path, n_ctx: int,
                   flash_attn: bool, type_k: str, type_v: str,
                   points: int, output: Path, reference: Path | None,
                   batch: int = 32, group_ids: list[str] | None = None) -> dict:
    validate_kv_config(flash_attn, type_k, type_v)
    from inference.adapters.llama_cpp_python import LlamaCppPythonMappingAdapter

    layout = inspect_gguf(model_path, n_ctx=n_ctx)
    limit = context_limit(layout)
    if n_ctx > limit:
        raise ValueError(f"n_ctx={n_ctx} oltre limite GGUF={limit}.")
    arch = layout["metadata"]["general.architecture"]
    if group_ids:
        from .mapping_grouping import group_inventory, native_multi_group_trial
        inventory = {g["id"]: g for g in group_inventory(layout)
                     if g["eligible_for_trial"]}
        if len(group_ids) != len(set(group_ids)) or any(
            gid not in inventory for gid in group_ids
        ):
            raise ValueError("Gruppi qualità CPU non validi.")
        trial = native_multi_group_trial(
            layout, [inventory[gid] for gid in group_ids]
        )
        rules = trial["rules"]
    else:
        rules, _ = all_gpu_rules(layout)
    blocks = layout["metadata"][f"{arch}.block_count"]
    payload = corpus.read_bytes()
    if not payload or len(payload) > 10_000_000:
        raise ValueError("Corpus vuoto o maggiore di 10 MB.")
    sha = hashlib.sha256(payload).hexdigest()

    llm = None
    try:
        llm, _ = LlamaCppPythonMappingAdapter().load(
            model_path, n_gpu_layers=blocks + 1, rules=rules,
            n_ctx=n_ctx, n_batch=batch, n_ubatch=batch,
            offload_kqv=False, flash_attn=flash_attn,
            type_k=type_k, type_v=type_v,
        )
        ids = list(llm.tokenize(payload, add_bos=True, special=True))
        if len(ids) < n_ctx:
            raise ValueError(
                f"Il corpus ha {len(ids)} token; richiesti almeno {n_ctx}."
            )
        ids = ids[:n_ctx]
        digest = hashlib.sha256(
            np.asarray(ids, dtype="<i4").tobytes()
        ).hexdigest()
        positions = score_positions(n_ctx, points)
        logits = native_sample_logits(llm, ids, positions, batch)
        targets = [ids[pos + 1] for pos in positions]

        result = {
            "success": True, "quality_kv_location": "cpu",
            "n_ctx": n_ctx, "flash_attn": flash_attn,
            "type_k": type_k, "type_v": type_v, "points": len(positions),
            "host_group_ids": sorted(group_ids or []),
            "corpus_sha256": sha, "token_prefix_sha256": digest,
            "score_positions": positions,
            "target_tokens_sha256": hashlib.sha256(
                np.asarray(targets, dtype="<i4").tobytes()
            ).hexdigest(),
        }
        if reference is None:
            np.savez_compressed(
                output, logits=logits, targets=np.asarray(targets, dtype=np.int32),
                positions=np.asarray(positions, dtype=np.int32),
                token_prefix_sha256=np.asarray(digest),
                corpus_sha256=np.asarray(sha),
            )
            result["reference_saved"] = True
        else:
            with np.load(reference, allow_pickle=False) as data:
                if (str(data["token_prefix_sha256"]) != digest
                    or str(data["corpus_sha256"]) != sha
                    or data["positions"].tolist() != positions
                    or data["targets"].tolist() != targets):
                    raise ValueError("Riferimento FP16 non corrisponde al corpus/token.")
                result["quality"] = sampled_metrics(
                    data["logits"], logits, targets,
                )
        return result
    finally:
        if llm is not None:
            llm.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--n-ctx", type=int, required=True)
    parser.add_argument("--flash-attn", choices=("on", "off"), required=True)
    parser.add_argument("--type-k", choices=("f16", "q8_0", "q4_0"), required=True)
    parser.add_argument("--type-v", choices=("f16", "q8_0", "q4_0"), required=True)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--group-ids", default=None)
    args = parser.parse_args()
    try:
        if args.batch < 1 or args.points < 2:
            raise ValueError("batch o punti non validi.")
        group_ids = json.loads(args.group_ids) if args.group_ids else None
        if group_ids is not None and (not isinstance(group_ids, list)
                                      or any(not isinstance(x, str) for x in group_ids)):
            raise ValueError("Lista gruppi qualità malformata.")
        result = quality_worker(
            model_path=args.model_path, corpus=args.corpus, n_ctx=args.n_ctx,
            flash_attn=args.flash_attn == "on", type_k=args.type_k,
            type_v=args.type_v, points=args.points, output=args.output,
            reference=args.reference, batch=args.batch,
            group_ids=group_ids,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({
            "success": False, "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
