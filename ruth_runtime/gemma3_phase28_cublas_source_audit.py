"""Read-only, fail-closed source audit of pinned BF16/cuBLAS operand provenance.

This recognizes narrow control-flow patterns in the exact clean pinned
llama.cpp source. It does NOT validate a live CUDA dispatch, associate
per-kernel buffers, instrument GPU work, or change production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .dependencies import PROJECT_ROOT
from .install_plan import LLAMA_CPP_SUBMODULE_COMMIT, SOURCE_COMMIT
from .source_build_plan import SOURCE_RELATIVE_PATH, inspect_checkout

CUDA_REL = Path("ggml/src/ggml-cuda/ggml-cuda.cu")
CONVERT_REL = Path("ggml/src/ggml-cuda/convert.cu")


def _between(data: str, before: str, after: str) -> str:
    """Return narrowly bounded source excerpt; absence is fail-closed."""
    if data.count(before) != 1:
        return ""
    body=data.split(before, 1)[1]
    if after not in body:
        return ""
    return body.split(after, 1)[0]


def audit_cublas_patterns(cuda: str, conversions: str) -> dict:
    """Recognize pinned branch structure, not a general C++ parser."""
    trait = _between(
        cuda, "struct batched_mul_mat_traits<GGML_TYPE_BF16> {",
        "struct batched_mul_mat_traits<GGML_TYPE_F16> {")
    impl = _between(
        cuda, "static void ggml_cuda_mul_mat_cublas_impl(",
        "static void ggml_cuda_mul_mat_cublas(")
    dispatch = _between(
        cuda, "static void ggml_cuda_mul_mat_cublas(",
        "static bool ggml_cuda_should_fuse_mul_mat(")
    entry = _between(
        cuda, "static void ggml_cuda_mul_mat(",
        "static bool ggml_cuda_mul_mat_id_needs_sync(")
    conv = _between(
        conversions, "to_bf16_cuda_t ggml_get_to_bf16_cuda(",
        "to_fp16_cuda_t ggml_get_to_fp16_cuda(")
    conv_vector = _between(
        conversions, "static void convert_unary_cont_cuda(",
        "to_bf16_cuda_t ggml_get_to_bf16_cuda(")

    checks = {
        "bf16_trait_cublas_f32_compute": all(x in trait for x in (
            "CUBLAS_COMPUTE_32F","CUDA_R_16BF",
            "using cuda_type = nv_bfloat16;",
            "ggml_get_to_bf16_cuda(src_type)")),
        "src0_bf16_reuse_or_convert": all(x in impl for x in (
            "if (src0->type == compute_type)",
            "src0_ptr = (const cuda_t *) src0->data;",
            "src0_alloc.alloc(ggml_nelements(src0));",
            "traits::convert(src0->type)",
            "traits::convert_nc(src0->type)",
            "convert_func(src0->data, src0_alloc.get()",
            "src0_ptr = src0_alloc.get();")),
        "src1_bf16_reuse_or_convert": all(x in impl for x in (
            "if (src1->type == compute_type)",
            "src1_ptr = (const cuda_t *) src1->data;",
            "src1_alloc.alloc(ggml_nelements(src1));",
            "traits::convert(src1->type)",
            "traits::convert_nc(src1->type)",
            "convert_func(src1->data, src1_alloc.get()",
            "src1_ptr = src1_alloc.get();")),
        "src0_src1_temporary_allocations": all(x in impl for x in (
            "ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());",
            "ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());")),
        "bf16_output_f32_on_nvidia": all(x in impl for x in (
            "compute_type == GGML_TYPE_BF16",
            "prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc)",
            "dst_ptr = (char *) dst_ddf;",
            "cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;",
            "cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;")),
        "cublas_variants_distinguished": all(x in impl for x in (
            "cublasSgemm(","cublasGemmEx(",
            "cublasGemmStridedBatchedEx(","cublasGemmBatchedEx(",
            "compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1",
            "ne12 == 1 && ne13 == 1",
            "r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2",
            "k_compute_batched_ptrs<<<")),
        "bf16_gemm_uses_converted_operand_ptrs": all(x in impl for x in (
            "src0_ptr, cu_data_type_a",
            "src1_ptr, cu_data_type_b",
            "cu_compute_type,",
            "if (cu_data_type != CUDA_R_32F)",
            "to_fp32_cuda(dst_temp.get(), dst_ddf")),
        "dispatch_precision_overrides": all(x in dispatch for x in (
            "ggml_type compute_type = src0->type;",
            "dst->op_params[0] == GGML_PREC_F32",
            'getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE")',
            "ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(",
            "ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(",
            "ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(")),
        "matmul_other_dispatches_possible": all(x in entry for x in (
            "ggml_cuda_mul_mat_vec_f(",
            "ggml_cuda_mul_mat_f(",
            "ggml_cuda_mul_mat_cublas(")),
        "f32_to_bf16_convert_lookup": all(x in conv for x in (
            "case GGML_TYPE_F32:",
            "return convert_unary_cont_cuda<float>;")),
        "vec4_conversion_conditional_not_universal": all(x in conv_vector for x in (
            "k % 4 == 0",
            "alignof(cvt_vec4<src_t>)",
            "alignof(cvt_vec4<dst_t>)",
            "convert_unary_cont_vec4<src_t, dst_t><<<",
            "convert_unary_cuda<src_t>(")),
    }
    return {
        "recognized_source_control_flow_patterns":checks,
        "all_pinned_source_patterns_recognized":all(checks.values()),
        "source_pattern_audit_not_ast_or_runtime_proof":True,
        "source_src0_src1_branch_identity_verified":False,
        "observed_17184_cast_to_gemm_buffer_identity_verified":False,
        "ggml_operator_to_kernel_identity_verified":False,
        "individual_cuda_graph_buffer_dependencies_verified":False,
        "conversion_elimination_safe_verified":False,
        "production_qualified":False,
        "gpu_inference_executed":False,
    }


def inspect_pinned_cublas_source(
    *, root: Path=PROJECT_ROOT, source_dir: Path | None=None,
) -> dict:
    root=Path(root).resolve(strict=True)
    expected=(root / SOURCE_RELATIVE_PATH).resolve()
    source=(Path(source_dir).resolve() if source_dir is not None else expected)
    result={
        "status":"pinned_cublas_source_unverified",
        "source_expected_parent_commit":SOURCE_COMMIT,
        "source_expected_llama_cpp_commit":LLAMA_CPP_SUBMODULE_COMMIT,
        "source_checkout_verified":False,
        "source_control_flow_patterns_verified":False,
        "observed_17184_cast_to_gemm_buffer_identity_verified":False,
        "conversion_elimination_safe_verified":False,
        "production_qualified":False,
        "gpu_inference_executed":False,
    }
    # Reuse Ruth's own approved source pathname and exact revision checks.
    if source != expected or expected.is_symlink():
        result["reason"]="source_path_not_approved_or_symlinked"
        return result
    state,parent_commit,child_commit=inspect_checkout(
        expected,git_available=shutil.which("git") is not None)
    result.update({
        "source_checkout_status":state,
        "source_parent_commit_observed":parent_commit,
        "source_llama_cpp_commit_observed":child_commit,
    })
    if state != "verified":
        result["reason"]="source_checkout_not_clean_and_exactly_pinned"
        return result
    child=expected/"vendor"/"llama.cpp"
    src_cu=child/CUDA_REL
    src_conv=child/CONVERT_REL
    # Avoid loading unexpected files or following altered symbolic links.
    if (not src_cu.is_file() or not src_conv.is_file()
        or src_cu.is_symlink() or src_conv.is_symlink()):
        result["reason"]="required_pinned_cuda_source_missing_or_symlinked"
        return result
    for fp in (src_cu,src_conv):
        if fp.stat().st_size > 2*1024*1024:
            result["reason"]="pinned_cuda_source_file_oversized"
            return result
    patterns=audit_cublas_patterns(
        src_cu.read_text(encoding="utf-8"),
        src_conv.read_text(encoding="utf-8"))
    result.update(patterns)
    result["source_checkout_verified"]=True
    result["source_control_flow_patterns_verified"]=patterns[
        "all_pinned_source_patterns_recognized"]
    result["status"]=(
        "pinned_cublas_source_control_flow_patterns_verified"
        if result["source_control_flow_patterns_verified"] else
        "pinned_cublas_source_patterns_unverified")
    return result


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(
        description="Offline pinned llama.cpp CUDA/cuBLAS BF16 source audit")
    parser.add_argument(
        "--source-dir",type=Path,
        help="Ruth approved checkout .ruth-src/llama-cpp-python (default)")
    args=parser.parse_args(argv)
    try:
        report=inspect_pinned_cublas_source(source_dir=args.source_dir)
    except (OSError,ValueError,UnicodeError) as e:
        report={
            "status":"pinned_cublas_source_unverified",
            "reason":type(e).__name__+": "+str(e)[:180],
            "source_control_flow_patterns_verified":False,
            "observed_17184_cast_to_gemm_buffer_identity_verified":False,
            "conversion_elimination_safe_verified":False,
            "production_qualified":False,
            "gpu_inference_executed":False,
        }
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report.get("source_control_flow_patterns_verified") is True else 2


if __name__=="__main__":
    raise SystemExit(main())
