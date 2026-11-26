#!/usr/bin/env bash
# Step 8: one server start per configuration, measured by scripts/step8-client.py. Run from the repository root:
#   experiments/2026-11-vllm-27b-tp2/scripts/run-step8.sh [name ...]   (default: the four below)
set -euo pipefail
E=experiments/2026-11-vllm-27b-tp2
OUT=$E/runs/8-parallel
names=("$@"); (( ${#names[@]} )) || names=(tp2-seqs2 tp2-seqs1 tp2-seqs4 pp2-seqs4)
for n in "${names[@]}"; do
  case $n in
    tp2-seqs2) cfg=nvfp4-mtp4;;           # the recipe: --max-num-seqs 2
    tp2-seqs1) cfg=nvfp4-mtp4-seqs1;;
    tp2-seqs4) cfg=nvfp4-mtp4-seqs4;;
    pp2-seqs4) cfg=nvfp4-mtp4-pp2-seqs4;;     # failed to start: no KV memory on the last stage (journal step 8)
    pp2p39-seqs4) cfg=nvfp4-mtp4-pp2p39-seqs4;;   # failed: 262144 does not fit the first stage's KV memory
    pp2p39-len196k-seqs4) cfg=nvfp4-mtp4-pp2p39-len196k-seqs4;;   # failed: the second stage holds at most 115,808 tokens
    pp2p39-len98k-seqs4) cfg=nvfp4-mtp4-pp2p39-len98k-seqs4;;   # started, then the second stage ran out of memory on the first request
    pp2p41-len64k-seqs4) cfg=nvfp4-mtp4-pp2p41-len64k-seqs4;;
    *) echo "unknown $n" >&2; exit 1;;
  esac
  echo "==== $n ($cfg.args), $(date -u +%T)"
  BENCH_CLIENT=$E/scripts/step8-client.py scripts/bench-run.sh "$OUT" "$n" 3 -- "$E/scripts/vllm-serve.sh" "$E/configs/$cfg.args" \
    || echo "!! $n failed (exit $?)"
  if docker ps --format '{{.Names}}' | grep -qx vllm-bench; then
    echo "!! vllm-bench still running after $n; stopping it"; docker stop -t 15 vllm-bench > /dev/null
  fi
done
