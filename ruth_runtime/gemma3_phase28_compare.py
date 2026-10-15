"""Offline numeric comparison of two aligned Phase28 logits artifacts.

No GPU use. Enforces both artifacts belong to the same n_ctx and have
bit-identical teacher-forced token IDs before computing numerical distances.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .dependencies import PROJECT_ROOT
from .gemma3_logits_compare import compare_logits

_FILENAME = re.compile(r"^ctx-(256|512|1024|2048)_flash-(off|on)_k-(F16|BF16|F32)_v-(F16|BF16|F32)_[A-Za-z0-9_]+\.npz$")


def load_artifact(root: Path, user_path: str):
    import numpy as np

    root = Path(root).resolve(strict=True)
    folder = root / ".ruth-build/phase28/gemma-3-1b-it"
    p = Path(user_path)
    p = p if p.is_absolute() else root / p
    name = _FILENAME.fullmatch(p.name)
    if (name is None or p.is_symlink() or not p.is_file()
        or p.parent.resolve() != folder.resolve(strict=True)
        or not 0 < p.stat().st_size < 40_000_000):
        raise ValueError("Expected one local pinned Phase28 numerical artifact")
    from hashlib import sha256
    digest = sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    with np.load(p, allow_pickle=False) as bundle:
        if set(bundle.files) != {"logits","prompt_ids","decode_ids"}:
            raise ValueError("Unexpected local numeric artifact schema")
        logits = bundle["logits"]
        prompt = bundle["prompt_ids"]
        follow = bundle["decode_ids"]
    n_ctx = int(name.group(1))
    if (logits.dtype != np.float32 or logits.ndim != 2
        or logits.shape[0] != 2 or not 512 <= logits.shape[1] <= 300000
        or not np.isfinite(logits).all()
        or prompt.dtype != np.int32 or prompt.ndim != 1
        or not 1 <= len(prompt) < n_ctx - 16
        or follow.dtype != np.int32 or follow.shape != (16,)):
        raise ValueError("Invalid Phase28 pinned logits or token coordinates")
    return {"logits":logits, "prompt_ids":prompt, "decode_ids":follow,
            "sha256":digest.hexdigest(),"n_ctx":n_ctx,"name":p.name}


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="Offline Phase28 identical-workload logits comparison")
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--candidate",required=True)
    options=parser.parse_args(argv)
    try:
        a=load_artifact(PROJECT_ROOT,options.baseline)
        b=load_artifact(PROJECT_ROOT,options.candidate)
        if a["n_ctx"] != b["n_ctx"]:
            raise ValueError("Long-context comparisons require equal n_ctx")
        result=compare_logits(a,b)
        result["n_ctx"] = a["n_ctx"]
        result["positions"] = {"last_prefill":0,"last_decode":1}
        result["native_kernel_dispatch_verified"] = False
        result["production_qualified"] = False
    except (OSError,RuntimeError,ValueError,TypeError) as exc:
        print(json.dumps({"status":"phase28_comparison_refused",
                          "error":type(exc).__name__+": "+str(exc)[:200]},
                         sort_keys=True))
        return 2
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
