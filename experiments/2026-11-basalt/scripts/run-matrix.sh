#!/usr/bin/env bash
# Step 11 of this experiment: the two-GPU matrix of 2026-11-strata step 13, for Basalt, in one go.
# Runs from experiments/2026-11-basalt; every run through scripts/bench-run.sh with a config from configs/.
set -uo pipefail
cd "$(dirname "$0")/.."
R=../../scripts
B=$HOME/basalt/v0.5.7/src
export BENCH_ENGINE_LOG=$HOME/.cache/basalt/2d72cadc08663624/engine.log
run() {  # outdir label reps [env...]
  local out=$1 label=$2 reps=$3; shift 3
  echo "== $(date +%H:%M:%S) $out/$label"
  env "$@" $R/bench-run.sh "runs/$out" "$label" "$reps" -- env -C "$B" bin/basalt serve \
    --config "$PWD/configs/basalt-$label.json" --python .venv/bin/python 2>&1 |
    grep -E 'ready in|FAIL|rep [0-9]|job rep|x[0-9]+:|GPU[01]:|RSS' | cut -c1-230
}

# 11a. long context and the four-turn job (no drafting, as Strata's step 12/13c)
for m in 1gpu peer split; do
  run 11a-long $m-nodraft-128k 3 LONG_MODE=ttft BENCH_CLIENT=$R/long-bench.py
  run 11a-long $m-nodraft-128k-cache 3 LONG_MODE=job BENCH_CLIENT=$R/long-bench.py
done

# 11c. several requests at once: short prompts (decode) and the 4.3K-token prose prompt
for c in 1gpu-nodraft-par4 peer-nodraft-par4 split-nodraft-par4 peer-mtp-par4 split-mtp-par4; do
  run 11c-concurrency-short $c 3 BENCH_KINDS=short BENCH_MAX_TOKENS=512 BENCH_CONCURRENCY=1,2,4
done
for c in peer-nodraft-par8 split-nodraft-par8; do
  run 11c-concurrency-short $c 3 BENCH_KINDS=short BENCH_MAX_TOKENS=512 BENCH_CONCURRENCY=8
done
for c in 1gpu-nodraft-par4 peer-nodraft-par4 split-nodraft-par4; do
  run 11c-concurrency $c 3 BENCH_KINDS=prose BENCH_CONCURRENCY=1,2,4
done

# 11b. shares
for c in split-nodraft-share46 split-nodraft-share64 peer-nodraft-share11; do
  run 11b-shares $c 5
done
echo "== $(date +%H:%M:%S) matrix done"
