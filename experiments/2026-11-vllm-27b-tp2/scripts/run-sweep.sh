#!/usr/bin/env bash
# Step 5: one server start per k (0 = fp8-nomtp.args, k = fp8-mtp<k>.args), decode with scripts/sweep-client.py;
# decode at depth as well for the k values in DEPTH_KS (default "0 4"). Run from the repository root:
#   experiments/2026-11-vllm-27b-tp2/scripts/run-sweep.sh [k ...]      (default: 0 1 2 3 4 5 6 7)
set -euo pipefail
E=experiments/2026-11-vllm-27b-tp2
OUT=$E/runs/5-sweep
DEPTH_KS=${DEPTH_KS:-"0 4"}
ks=("$@"); (( ${#ks[@]} )) || ks=(0 1 2 3 4 5 6 7)
for k in "${ks[@]}"; do
  cfg=$E/configs/fp8-mtp$k.args; [ "$k" = 0 ] && cfg=$E/configs/fp8-nomtp.args
  depth=0; [[ " $DEPTH_KS " == *" $k "* ]] && depth=1
  echo "==== k=$k ($cfg), depth: $depth, $(date -u +%T)"
  SWEEP_DEPTH=$depth BENCH_CLIENT=$E/scripts/sweep-client.py \
    scripts/bench-run.sh "$OUT" "k$k" 3 -- "$E/scripts/vllm-serve.sh" "$cfg"
  if docker ps --format '{{.Names}}' | grep -qx vllm-bench; then
    echo "!! vllm-bench still running after k=$k; stopping it"; docker stop -t 15 vllm-bench > /dev/null
  fi
done
