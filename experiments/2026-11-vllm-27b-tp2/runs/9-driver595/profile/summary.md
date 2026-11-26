| rank | span ms | busy ms | NCCL kernels | NCCL ms | NCCL / span | NCCL mean µs | max µs |
|---|---|---|---|---|---|---|---|
| rank0 | 1299.3 | 1276.5 | 260 | 725.7 | 55.8 % | 2791 | 5412 |
| rank1 | 1299.4 | 1290.3 | 260 | 742.8 | 57.2 % | 2857 | 5516 |

**rank0: top kernels by total time**

| kernel | count | ms | share of span |
|---|---|---|---|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage` | 258 | 725.4 | 55.8 % |
| `_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tuple` | 514 | 309.3 | 23.8 % |
| `void flashinfer::BatchPrefillWithPagedKVCacheKernel` | 32 | 58.3 | 4.5 % |
| `kernel_cutlass_kernel_flashinfergdn_kernelsdelta_rule_dsldelta_rule_sm` | 96 | 25.9 | 2.0 % |
| `void vllm::silu_mul_cvt_fp16_to_fp4` | 128 | 25.2 | 1.9 % |
| `void vllm::cvt_fp16_to_fp4` | 386 | 19.1 | 1.5 % |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_1` | 64 | 16.4 | 1.3 % |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_scaled_fp4_quant_0` | 96 | 14.6 | 1.1 % |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_scaled_fp4_quant_1` | 64 | 13.1 | 1.0 % |
| `_causal_conv1d_fwd_kernel` | 96 | 10.7 | 0.8 % |
| `_fused_post_conv_kernel` | 96 | 10.3 | 0.8 % |
| `void at::native::elementwise_kernel` | 384 | 8.6 | 0.7 % |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_2` | 32 | 8.0 | 0.6 % |
| `layer_norm_fwd_kernel` | 96 | 7.6 | 0.6 % |

**rank1: top kernels by total time**

| kernel | count | ms | share of span |
|---|---|---|---|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage` | 258 | 742.7 | 57.2 % |
| `_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tuple` | 514 | 306.6 | 23.6 % |
| `void flashinfer::BatchPrefillWithPagedKVCacheKernel` | 32 | 57.9 | 4.5 % |
| `kernel_cutlass_kernel_flashinfergdn_kernelsdelta_rule_dsldelta_rule_sm` | 96 | 25.7 | 2.0 % |
| `void vllm::silu_mul_cvt_fp16_to_fp4` | 128 | 25.3 | 1.9 % |
| `void vllm::cvt_fp16_to_fp4` | 386 | 19.2 | 1.5 % |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_1` | 64 | 16.4 | 1.3 % |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_scaled_fp4_quant_0` | 96 | 14.6 | 1.1 % |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_scaled_fp4_quant_1` | 64 | 13.1 | 1.0 % |
| `_causal_conv1d_fwd_kernel` | 96 | 10.7 | 0.8 % |
| `_fused_post_conv_kernel` | 96 | 10.3 | 0.8 % |
| `void at::native::elementwise_kernel` | 384 | 8.6 | 0.7 % |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_2` | 32 | 8.1 | 0.6 % |
| `layer_norm_fwd_kernel` | 96 | 7.6 | 0.6 % |
