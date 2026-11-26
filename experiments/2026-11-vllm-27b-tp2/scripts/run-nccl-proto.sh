#!/usr/bin/env bash
# Step 9g: which NCCL protocol and algorithm suits the prefill's 40 MiB all-reduce (the profile shows RING_LL)?
#
#   experiments/2026-11-vllm-27b-tp2/scripts/run-nccl-proto.sh OUTDIR [IMAGE]
#
# all_reduce_perf at 10 KiB, 50 KiB, 1 MiB, 40 MiB under NCCL_PROTO = (unset), LL, LL128, Simple, with NCCL_P2P_DISABLE=1
# as the vLLM configs; with IMAGE, inside that container (same mounts as run-nccl-tests.sh). The log line "Algo Ring proto"
# of NCCL_DEBUG=INFO,TUNING is not needed: the chosen kernel shows in the nccl-tests profile; NCCL_PROTO is forced.
set -euo pipefail
out=${1:?output directory}; image=${2:-}
bin=$HOME/nccl-tests/current/src/build/all_reduce_perf
mkdir -p "$out"; tag=host; [[ -n $image ]] && tag=container-${image##*:}
f=$out/proto-$tag.out; : > "$f"
{ echo "date: $(date)"; echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"; echo "image: ${image:-none (host)}"; } > "$out/proto-$tag.cmd"
for proto in "" LL LL128 Simple; do
  for size in 10K 50K 1M 40M; do
    echo "### NCCL_PROTO=${proto:-unset} -b $size" >> "$f"
    cmd=(-b $size -e $size -g 2 -d bfloat16 -o sum -w 20 -n 100 -c 1)
    echo "NCCL_PROTO=${proto:-unset} NCCL_P2P_DISABLE=1 all_reduce_perf ${cmd[*]}" >> "$out/proto-$tag.cmd"
    if [[ -z $image ]]; then
      env NCCL_P2P_DISABLE=1 ${proto:+NCCL_PROTO=$proto} "$bin" "${cmd[@]}" >> "$f" 2>&1
    else
      docker run --rm --gpus all --ipc host --entrypoint "/nccl-tests/$(basename "$(readlink -f "$HOME/nccl-tests/current")")/src/build/all_reduce_perf" \
        -v "$HOME/nccl-tests":/nccl-tests:ro \
        -e LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib \
        -e NCCL_P2P_DISABLE=1 ${proto:+-e NCCL_PROTO=$proto} "$image" "${cmd[@]}" >> "$f" 2>&1
    fi
  done
done
awk '/^### /{h=$0} /^ +[0-9]+ +[0-9]+ +bfloat16/{print h" | "$6" us "$8" GB/s"}' "$f"
