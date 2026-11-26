#!/usr/bin/env bash
# Step 9f: NCCL all-reduce between the two cards at the message sizes of the 27B's tensor-parallel prefill and decode.
#
#   experiments/2026-11-vllm-27b-tp2/scripts/run-nccl-tests.sh OUTDIR [IMAGE]
#
# With IMAGE (e.g. local/vllm-openai:v0.30.0) the same binary runs inside that container, loading the image's own
# NCCL (torch's nvidia/nccl) and CUDA runtime instead of the host's; output files get the image tag in their names.
#
# Needs ~/nccl-tests/current (scripts/build-nccl-tests.sh). One process drives both GPUs (-g 2), bf16, sum.
# Sizes: a sweep 8 KB .. 128 MB (x2), then exactly 40 MiB (a 4,096-token prefill chunk: 4096 x 5120 x 2 bytes) and
# 10 KiB / 50 KiB (one decode token / 5 tokens with MTP k=4: 5120 x 2 bytes each). Two settings, as the vLLM runs:
# NCCL_P2P_DISABLE=1 (the configs) and 0 (the NCCL default), each with the NCCL transport logged.
set -euo pipefail
out=${1:?output directory}
image=${2:-}
bin=$HOME/nccl-tests/current/src/build/all_reduce_perf
[[ -x $bin ]] || { echo "missing $bin" >&2; exit 1; }
mkdir -p "$out"
tag=${image##*:}
[[ -n $image ]] && tag=container-$tag || tag=host
run_bin() {  # args: nccl-tests arguments
  if [[ -z $image ]]; then "$bin" "$@"; else
    docker run --rm --gpus all --ipc host --entrypoint /nccl-tests/$(basename "$(readlink -f "$HOME/nccl-tests/current")")/src/build/all_reduce_perf \
      -v "$HOME/nccl-tests":/nccl-tests:ro \
      -e LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib \
      -e NCCL_P2P_DISABLE -e NCCL_DEBUG -e NCCL_DEBUG_SUBSYS "$image" "$@"
  fi
}
for p2p in 1 0; do
  f=$out/allreduce-$tag-p2pdisable$p2p
  {
    echo "date: $(date)"
    echo "kernel: $(uname -r)"
    echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit --format=csv,noheader | sed 's/^/gpu: /'
    echo "build: $(tr '\n' ';' < "$HOME/nccl-tests/current/build-info.txt")"
    echo "image: ${image:-none (host)}"
    echo "env: NCCL_P2P_DISABLE=$p2p NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM"
  } > "$f.cmd"
  export NCCL_P2P_DISABLE=$p2p NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM
  for sizes in "-b 8K -e 128M -f 2" "-b 10K -e 10K" "-b 50K -e 50K" "-b 40M -e 40M"; do
    echo "command: $bin $sizes -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1" | tee -a "$f.cmd"
    echo "### $sizes" >> "$f.out"
    run_bin $sizes -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1 >> "$f.out" 2>&1
  done
done
