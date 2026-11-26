date: Thu Nov 26 08:18:52 PM UTC 2026
driver: 595.91.07
image: local/vllm-openai:v0.30.0
NCCL_PROTO=unset NCCL_P2P_DISABLE=1 all_reduce_perf -b 10K -e 10K -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=unset NCCL_P2P_DISABLE=1 all_reduce_perf -b 50K -e 50K -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=unset NCCL_P2P_DISABLE=1 all_reduce_perf -b 1M -e 1M -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=unset NCCL_P2P_DISABLE=1 all_reduce_perf -b 40M -e 40M -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=LL NCCL_P2P_DISABLE=1 all_reduce_perf -b 10K -e 10K -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=LL NCCL_P2P_DISABLE=1 all_reduce_perf -b 50K -e 50K -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=LL NCCL_P2P_DISABLE=1 all_reduce_perf -b 1M -e 1M -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=LL NCCL_P2P_DISABLE=1 all_reduce_perf -b 40M -e 40M -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=LL128 NCCL_P2P_DISABLE=1 all_reduce_perf -b 10K -e 10K -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=LL128 NCCL_P2P_DISABLE=1 all_reduce_perf -b 50K -e 50K -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=LL128 NCCL_P2P_DISABLE=1 all_reduce_perf -b 1M -e 1M -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=LL128 NCCL_P2P_DISABLE=1 all_reduce_perf -b 40M -e 40M -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=Simple NCCL_P2P_DISABLE=1 all_reduce_perf -b 10K -e 10K -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=Simple NCCL_P2P_DISABLE=1 all_reduce_perf -b 50K -e 50K -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=Simple NCCL_P2P_DISABLE=1 all_reduce_perf -b 1M -e 1M -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
NCCL_PROTO=Simple NCCL_P2P_DISABLE=1 all_reduce_perf -b 40M -e 40M -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
