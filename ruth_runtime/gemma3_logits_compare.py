"""Offline, full-vocabulary numerical comparison of Phase27 logits.

Does not import llama_cpp, initialize CUDA, download a model, or select the
reference precision. Uses aligned, teacher-forced token streams only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .dependencies import PROJECT_ROOT

PHASE27_DIR = Path(".ruth-build/phase27/gemma-3-1b-it")


def read_logits(root: Path, supplied_path: str):
    import numpy as np

    root = Path(root).resolve(strict=True)
    base = root / PHASE27_DIR
    file = Path(supplied_path)
    if not file.is_absolute():
        file = root / file
    if (file.is_symlink() or not file.is_file()
        or file.suffix != ".npz"
        or not file.parent.resolve() == base.resolve(strict=True)
        or file.stat().st_size > 150_000_000):
        raise ValueError("Not a bounded local pinned Phase27 logits artifact")
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    with np.load(file, allow_pickle=False) as z:
        if set(z.files) != {"logits", "prompt_ids", "decode_ids"}:
            raise ValueError("Unexpected numeric artifact fields")
        logits = z["logits"]
        prompt_ids = z["prompt_ids"]
        decode_ids = z["decode_ids"]
    if (logits.dtype != np.float32 or logits.ndim != 2
        or logits.shape[0] != 7
        or not 512 <= logits.shape[1] <= 300000
        or not np.isfinite(logits).all()
        or prompt_ids.dtype != np.int32 or prompt_ids.ndim != 1
        or not 1 <= len(prompt_ids) <= 64
        or decode_ids.dtype != np.int32 or decode_ids.shape != (6,)):
        raise ValueError("Malformed or nonfinite native logits")
    return {
        "logits": logits, "prompt_ids": prompt_ids, "decode_ids": decode_ids,
        "sha256": digest.hexdigest(), "name": file.name,
    }


def compare_logits(baseline: Any, candidate: Any) -> dict:
    import numpy as np
    reference = baseline["logits"]
    other = candidate["logits"]
    if (reference.shape != other.shape
        or not np.array_equal(baseline["prompt_ids"], candidate["prompt_ids"])
        or not np.array_equal(baseline["decode_ids"], candidate["decode_ids"])):
        raise ValueError("Tokenization and full vocabulary must match exactly")
    # Float64, shifted log-sum-exp avoids underflow and large-logit overflow.
    base = reference.astype(np.float64)
    test = other.astype(np.float64)
    def probs_logits(logits):
        peak = np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits - peak)
        log_z = np.log(np.sum(exp, axis=1, keepdims=True)) + peak
        return np.exp(logits - log_z), logits - log_z
    p, lp = probs_logits(base)
    q, lq = probs_logits(test)
    # Avoid 0 * -inf: both source logsoftmax vectors are finite.
    kl_pq = np.sum(p * (lp - lq), axis=1)
    kl_qp = np.sum(q * (lq - lp), axis=1)
    m = (p + q) / 2.0
    log_m = np.log(np.maximum(m, np.finfo(np.float64).tiny))
    js = .5 * np.sum(p * (lp - log_m), axis=1) + (
         .5 * np.sum(q * (lq - log_m), axis=1))
    diff = test - base
    return {
        "status": "full_logits_numeric_comparison",
        "baseline_artifact_sha256": baseline["sha256"],
        "candidate_artifact_sha256": candidate["sha256"],
        "vocabulary": int(reference.shape[1]),
        "positions": int(reference.shape[0]),
        "same_teacher_forced_tokens": True,
        "max_absolute_logit_difference": float(np.max(np.abs(diff))),
        "mean_absolute_logit_difference": float(np.mean(np.abs(diff))),
        "root_mean_square_logit_difference": float(np.sqrt(np.mean(diff*diff))),
        "top1_match_fraction": float(np.mean(np.argmax(base, axis=1) ==
                                              np.argmax(test, axis=1))),
        "kl_baseline_to_candidate_mean_nats": float(np.mean(np.maximum(kl_pq, 0))),
        "kl_candidate_to_baseline_mean_nats": float(np.mean(np.maximum(kl_qp, 0))),
        "jensen_shannon_mean_nats": float(np.mean(np.maximum(js, 0))),
        "per_position": [{
            "index": int(i), "stage": "prefill" if i == 0 else "decode",
            "max_absolute_logit_difference": float(np.max(np.abs(diff[i]))),
            "top1_equal": bool(np.argmax(base[i]) == np.argmax(test[i])),
            "kl_baseline_to_candidate_nats": float(max(kl_pq[i], 0)),
        } for i in range(len(base))],
        "native_kernel_dispatch_verified": False,
        "production_qualified": False,
        "reference_precision_is_ground_truth": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline Phase27 full-logit distance")
    p.add_argument("--baseline", required=True,
                   help="Local Phase27 .npz path to use as comparison baseline")
    p.add_argument("--candidate", required=True,
                   help="Local Phase27 .npz path to compare")
    opts = p.parse_args(argv)
    try:
        a = read_logits(PROJECT_ROOT, opts.baseline)
        b = read_logits(PROJECT_ROOT, opts.candidate)
        result = compare_logits(a, b)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        print(json.dumps({
            "status": "numerical_comparison_refused",
            "error": type(exc).__name__ + ": " + str(exc)[:240],
        }, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
