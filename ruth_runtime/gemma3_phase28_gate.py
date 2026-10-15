"""Strict, allocation-and-inference gate for isolated Phase28 experiments.

These are *new* experimental contexts; Phase26/27 pinned gates remain untouched.
No production promotion, retry, download or automatic run is authorized here.
"""
from __future__ import annotations

from dataclasses import dataclass

# Each context has an explicit prompt length; the remaining space supports
# fixed 16-token decode without wrapping/sliding by lack of capacity.
LENGTHS = ((256, 128), (512, 384), (1024, 768), (2048, 1536))
DECODE_STEPS = 16
N_BATCH = 64
N_UBATCH = 64
WARMUPS = 1
REPEATS = 3

# Distinct, source-motivated choices; no automatic optimization.
CANDIDATES = (
    ("f16-off", "off", "F16", "F16"),
    ("bf16-on", "on", "BF16", "BF16"),
    ("f32-off", "off", "F32", "F32"),
    ("mixed-on", "on", "BF16", "F16"),
)


@dataclass(frozen=True)
class Phase28Plan:
    candidate: str
    flash_attention: str
    type_k: str
    type_v: str
    n_ctx: int
    prompt_tokens: int
    decode_tokens: int = DECODE_STEPS
    n_batch: int = N_BATCH
    n_ubatch: int = N_UBATCH
    swa_full: bool = True

    def as_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "flash_attention": self.flash_attention, "type_k": self.type_k,
            "type_v": self.type_v, "n_ctx": self.n_ctx,
            "prompt_tokens": self.prompt_tokens,
            "decode_tokens": self.decode_tokens,
            "n_batch": self.n_batch, "n_ubatch": self.n_ubatch,
            "swa_full": self.swa_full,
            "experimental_inference_only": True,
            "production_qualified": False,
        }


def plans() -> tuple[Phase28Plan, ...]:
    return tuple(
        Phase28Plan(name, flash, k, v, ctx, length)
        for ctx, length in LENGTHS
        for name, flash, k, v in CANDIDATES
    )


def verify_plan(candidate: str, n_ctx: int) -> Phase28Plan:
    if type(candidate) is not str or type(n_ctx) is not int:
        raise ValueError("Phase28 requires exact candidate and integer context")
    matches = [x for x in plans()
               if x.candidate == candidate and x.n_ctx == n_ctx]
    if len(matches) != 1:
        raise ValueError("Unapproved Phase28 pinned context/candidate pair")
    plan = matches[0]
    if not (0 < plan.prompt_tokens and
            plan.prompt_tokens + plan.decode_tokens < plan.n_ctx and
            plan.n_ubatch <= plan.n_batch <= plan.n_ctx):
        raise RuntimeError("Phase28 internal token/context guard mismatch")
    return plan


def theoretical_kv_mib(plan: Phase28Plan) -> float:
    """Logical only: 26 layers * n_ctx * KV head 256, each for K and V.

    Does not estimate compute transients, CUDA driver costs or full VRAM.
    """
    sizes = {"F16": 2, "BF16": 2, "F32": 4}
    return 26 * plan.n_ctx * 256 * (sizes[plan.type_k] + sizes[plan.type_v]) / 1048576
