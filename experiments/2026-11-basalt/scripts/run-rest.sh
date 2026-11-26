#!/usr/bin/env bash
# Steps 8, 9 and 10 of this experiment, and a control run, in one go. Runs from experiments/2026-11-basalt.
set -uo pipefail
cd "$(dirname "$0")/.."
R=../../scripts
B=$HOME/basalt/v0.5.7/src
run() {  # outdir label reps [env...]
  local out=$1 label=$2 reps=$3; shift 3
  echo "== $(date +%H:%M:%S) $out/$label"
  env "$@" $R/bench-run.sh "runs/$out" "$label" "$reps" -- env -C "$B" bin/basalt serve \
    --config "$PWD/configs/basalt-$label.json" --python .venv/bin/python 2>&1 |
    grep -E 'ready in|FAIL|rep [1-5] |rep [12] |GPU[01]:|RSS' | cut -c1-200
}
GSQ=$HOME/.cache/basalt/2d72cadc08663624/engine.log

# control: step 6's one-GPU run again, with no download running
run 6-7-modes-control 1gpu-nodraft 5 BENCH_ENGINE_LOG=$GSQ

# 8. route tail skip at rank 7 (Strata's step 9)
run 8-skip 1gpu-nodraft-skip7 5 BENCH_ENGINE_LOG=$GSQ
run 8-skip peer-mtp-skip7 5 BENCH_ENGINE_LOG=$GSQ

# 9. the UD-IQ3_XXS pack; its engine log is another file (another model id), found after the first start
for c in ud-1gpu-nodraft ud-1gpu-mtp ud-peer-nodraft ud-peer-mtp ud-split-mtp; do
  UDLOG=$(ls -t "$HOME"/.cache/basalt/*/engine.log 2>/dev/null | grep -v 2d72cadc08663624 | head -1)
  run 9-ud $c 5 ${UDLOG:+BENCH_ENGINE_LOG=$UDLOG}
done

# 10. quality: the ten prompts, twice each
for c in 1gpu-nodraft 1gpu-nodraft-det 1gpu-nodraft-det-skip7 peer-mtp peer-mtp-det; do
  run 10-quality $c 2 BENCH_CLIENT=$R/quality-check.py BENCH_ENGINE_LOG=$GSQ
done
echo "== $(date +%H:%M:%S) done"
