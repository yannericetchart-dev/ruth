"""No-GPU source-pattern regression tests; temporary fake checkout only."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ruth_runtime.gemma3_phase28_cublas_source_audit import (
    audit_cublas_patterns, inspect_pinned_cublas_source,
)
from ruth_runtime.source_build_plan import SOURCE_RELATIVE_PATH


def synthetic_sources() -> tuple[str,str]:
    cuda="""
    struct batched_mul_mat_traits<GGML_TYPE_BF16> {
        using cuda_type = nv_bfloat16;
        static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
        static inline const cudaDataType_t data_type = CUDA_R_16BF;
        static inline auto convert(ggml_type src_type) {
            return ggml_get_to_bf16_cuda(src_type);
        }
    };
    struct batched_mul_mat_traits<GGML_TYPE_F16> {
    };

    static void ggml_cuda_mul_mat_cublas_impl(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, ggml_tensor * dst) {
        ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
        ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());
        if (src0->type == compute_type) {
            src0_ptr = (const cuda_t *) src0->data;
        } else {
            src0_alloc.alloc(ggml_nelements(src0));
            traits::convert(src0->type);
            traits::convert_nc(src0->type);
            convert_func(src0->data, src0_alloc.get(), n, main_stream);
            src0_ptr = src0_alloc.get();
        }
        if (src1->type == compute_type) {
            src1_ptr = (const cuda_t *) src1->data;
        } else {
            src1_alloc.alloc(ggml_nelements(src1));
            traits::convert(src1->type);
            traits::convert_nc(src1->type);
            convert_func(src1->data, src1_alloc.get(), n, main_stream);
            src1_ptr = src1_alloc.get();
        }
        if (compute_type == GGML_TYPE_BF16) {
            prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc);
        }
        if (prefer_f32_output) {
            dst_ptr = (char *) dst_ddf;
            cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
            cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        }
        if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
            cublasSgemm(...);
        } else if (ne12 == 1 && ne13 == 1) {
            cublasGemmEx(src0_ptr, cu_data_type_a,
                         src1_ptr, cu_data_type_b,
                         cu_compute_type, ...);
        } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            cublasGemmStridedBatchedEx(...);
        } else {
            k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(...);
            cublasGemmBatchedEx(...);
        }
        if (cu_data_type != CUDA_R_32F) {
            to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
        }
    }

    static void ggml_cuda_mul_mat_cublas(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, ggml_tensor * dst) {
        ggml_type compute_type = src0->type;
        if (dst->op_params[0] == GGML_PREC_F32) {
            compute_type = GGML_TYPE_F32;
        }
        const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
        ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(...);
        ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(...);
        ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(...);
    }

    static bool ggml_cuda_should_fuse_mul_mat(...) {
    }

    static void ggml_cuda_mul_mat(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, ggml_tensor * dst) {
        ggml_cuda_mul_mat_vec_f(...);
        ggml_cuda_mul_mat_f(...);
        ggml_cuda_mul_mat_cublas(...);
    }

    static bool ggml_cuda_mul_mat_id_needs_sync(...) {
    }
    """
    conv="""
    static void convert_unary_cont_cuda(
        const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
        if (k % 4 == 0 &&
            (uintptr_t) vx % alignof(cvt_vec4<src_t>) == 0 &&
            (uintptr_t) y % alignof(cvt_vec4<dst_t>) == 0) {
            convert_unary_cont_vec4<src_t, dst_t><<<blocks,threads>>>(...);
            return;
        }
        convert_unary_cuda<src_t>(...);
    }
    to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
        switch (type) {
            case GGML_TYPE_F32:
                return convert_unary_cont_cuda<float>;
            default: return nullptr;
        }
    }
    to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    }
    """
    return cuda,conv


class TestCublasPinnedSourceAudit(unittest.TestCase):
    def test_branch_patterns_found_but_no_per_kernel_buffer_identity(self):
        cu,conv=synthetic_sources()
        out=audit_cublas_patterns(cu,conv)
        self.assertTrue(out["all_pinned_source_patterns_recognized"])
        self.assertEqual(
            len(out["recognized_source_control_flow_patterns"]),10)
        self.assertTrue(all(
            out["recognized_source_control_flow_patterns"].values()))
        self.assertFalse(
            out["observed_17184_cast_to_gemm_buffer_identity_verified"])
        self.assertFalse(out["conversion_elimination_safe_verified"])
        self.assertFalse(out["production_qualified"])

    def test_missing_conversion_or_forced_f32_path_fails_closed(self):
        cu,conv=synthetic_sources()
        no_convert=conv.replace(
            "return convert_unary_cont_cuda<float>;",
            "return nullptr;")
        r=audit_cublas_patterns(cu,no_convert)
        self.assertFalse(r["all_pinned_source_patterns_recognized"])
        self.assertFalse(r["recognized_source_control_flow_patterns"][
            "f32_to_bf16_convert_lookup"])
        r=audit_cublas_patterns(
            cu.replace('getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE")',
                       'getenv("NEW_CUDA_PARAMETER")'),conv)
        self.assertFalse(r["recognized_source_control_flow_patterns"][
            "dispatch_precision_overrides"])

    def test_missing_dirty_or_mismatched_source_cannot_verify(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/SOURCE_RELATIVE_PATH
            source.mkdir(parents=True)
            with patch(
                "ruth_runtime.gemma3_phase28_cublas_source_audit.inspect_checkout",
                return_value=("dirty","parent","child")
            ):
                result=inspect_pinned_cublas_source(root=root)
            self.assertEqual(result["status"],
                             "pinned_cublas_source_unverified")
            self.assertFalse(result["source_control_flow_patterns_verified"])
            self.assertFalse(result["conversion_elimination_safe_verified"])

    def test_approved_pinned_checkout_pattern_scope_only(self):
        cu,conv=synthetic_sources()
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/SOURCE_RELATIVE_PATH
            folder=source/"vendor/llama.cpp/ggml/src/ggml-cuda"
            folder.mkdir(parents=True)
            (folder/"ggml-cuda.cu").write_text(cu)
            (folder/"convert.cu").write_text(conv)
            with patch(
                "ruth_runtime.gemma3_phase28_cublas_source_audit.inspect_checkout",
                return_value=("verified","parent","child")
            ):
                r=inspect_pinned_cublas_source(root=root)
            self.assertEqual(
                r["status"],
                "pinned_cublas_source_control_flow_patterns_verified")
            self.assertTrue(r["source_checkout_verified"])
            self.assertFalse(
                r["observed_17184_cast_to_gemm_buffer_identity_verified"])
            self.assertFalse(r["gpu_inference_executed"])


if __name__=="__main__":
    unittest.main()
