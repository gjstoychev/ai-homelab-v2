date: Thu Nov 26 07:40:12 PM UTC 2026
kernel: 7.0.0-44-generic
driver: 595.91.07
gpu: 0, NVIDIA GeForce RTX 5060 Ti, 00000000:01:00.0, 180.00 W
gpu: 1, NVIDIA GeForce RTX 5060 Ti, 00000000:02:00.0, 180.00 W
build: nccl-tests v2.22.0, commit 826b3e6, built 2026-11-26T19:39:39Z;nvcc: Build cuda_13.0.r13.0/compiler.36424714_0;nccl: 2.28.9-1+cuda13.0;
env: NCCL_P2P_DISABLE=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM
command: /home/llama/nccl-tests/current/src/build/all_reduce_perf -b 8K -e 128M -f 2 -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
command: /home/llama/nccl-tests/current/src/build/all_reduce_perf -b 10K -e 10K -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
command: /home/llama/nccl-tests/current/src/build/all_reduce_perf -b 50K -e 50K -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
command: /home/llama/nccl-tests/current/src/build/all_reduce_perf -b 40M -e 40M -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1
