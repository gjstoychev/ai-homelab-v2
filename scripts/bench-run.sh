#!/usr/bin/env bash
# Start one server configuration, measure it with scripts/oai-bench.py, stop it, and keep everything.
#
#   scripts/bench-run.sh OUTDIR LABEL [REPS] -- <server command ...>
#   (REPS 0: start, report the load, stop; no requests)
#   e.g. scripts/bench-run.sh experiments/2026-11-strata/runs/7-llamacpp 1gpu-fit 5 -- \
#          env CUDA_VISIBLE_DEVICES=0 ~/llama.cpp/v1.0.0/build/bin/llama-server -m <shard 1> --port 8080 ...
#
# The server must listen on 127.0.0.1:8080 and answer /health with 200 when it is ready (llama-server and Strata's
# serve/server.py both do). In OUTDIR/LABEL.*:
#   .cmd      the command, date, driver, GPUs, and whether the page cache was dropped
#   .out      the server's output
#   .gpu.csv  nvidia-smi every 0.5 s (power, memory, clocks, temperature, utilisation)
#   .ram.csv  every 1 s: MemAvailable, page cache, RSS of the server's process group, MiB read from the NVMe
#   .cpu.csv  CPU package power every 1 s (RAPL), only when sudo works without a password prompt
# and one line per request in OUTDIR/results.jsonl.
# With BENCH_SUDO=1 (the user's own shell, after `sudo -v`) the page cache is dropped first and CPU power is
# sampled with sudo; without it both are skipped and .cmd says so. Claude Code runs it without BENCH_SUDO: sudo is
# denied to it by the managed settings, and a cached sudo ticket of the user's must not be used in its place.
set -euo pipefail

OUT=${1:?usage: $0 OUTDIR LABEL [REPS] -- server command}
LABEL=${2:?usage: $0 OUTDIR LABEL [REPS] -- server command}
shift 2
REPS=5
if [ $# -gt 0 ] && [ "$1" != "--" ]; then REPS=$1; shift; fi
[ "${1:-}" = "--" ] && shift
(( $# )) || { echo "no server command" >&2; exit 1; }
CMD=("$@")

here=$(cd "$(dirname "$0")" && pwd)
URL=http://127.0.0.1:8080
mkdir -p "$OUT"
P=$OUT/$LABEL
[ ! -e "$P.out" ] || { echo "ABORT: $P.out exists; this run was done already" >&2; exit 1; }

[ "$(systemctl is-active ollama || true)" = inactive ] || { echo "ABORT: ollama is not inactive" >&2; exit 1; }
apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
[ -z "$apps" ] || { echo "ABORT: processes on the GPUs: $apps" >&2; exit 1; }
! curl -fs "$URL/health" > /dev/null 2>&1 || { echo "ABORT: something answers on $URL" >&2; exit 1; }

SUDO=0; [ "${BENCH_SUDO:-0}" = 1 ] && sudo -n true 2>/dev/null && SUDO=1
{
  echo "date: $(date -u)"
  echo "label: $LABEL, reps: $REPS"
  echo "kernel: $(uname -r)"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit --format=csv,noheader | sed 's/^/gpu: /'
  echo "page cache dropped: $([ $SUDO = 1 ] && echo yes || echo 'no (BENCH_SUDO not set)')"
  echo "cpu power: $([ $SUDO = 1 ] && echo RAPL || echo 'not measured (BENCH_SUDO not set)')"
  echo "command: ${CMD[*]}"
} | tee "$P.cmd"
# a config file named in the command is copied next to the run
for arg in "${CMD[@]}"; do [[ $arg == *.json && -f $arg ]] && cp "$arg" "$P.config.json"; done

if [ $SUDO = 1 ]; then sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null; fi

Q=timestamp,index,pstate,power.draw,temperature.gpu,memory.used,utilization.gpu,clocks.gr,clocks.mem
nvidia-smi --query-gpu=$Q --format=csv -lms 500 > "$P.gpu.csv" &
GPU_SAMPLER=$!
CPU_SAMPLER=
if [ $SUDO = 1 ]; then
  sudo -n "$here/../experiments/2026-11-llamacpp-flags/scripts/cpu-power.sh" "$P.cpu.csv" $$ &
  CPU_SAMPLER=$!
fi
SERVER=
cleanup() {
  set +e
  if [ -n "$SERVER" ]; then
    kill -TERM -- "-$SERVER" 2>/dev/null                  # the whole process group (Strata: server.py and its engine)
    for _ in $(seq 60); do kill -0 -- "-$SERVER" 2>/dev/null || break; sleep 0.5; done
    kill -KILL -- "-$SERVER" 2>/dev/null
  fi
  kill "$GPU_SAMPLER" "${RAM_SAMPLER:-}" 2>/dev/null
  [ -z "$CPU_SAMPLER" ] || { sudo -n kill "$CPU_SAMPLER" 2>/dev/null; sudo -n chown "$(id -u):$(id -g)" "$P.cpu.csv" 2>/dev/null; }
}
trap cleanup EXIT

t0=$(date +%s.%N)
setsid "${CMD[@]}" > "$P.out" 2>&1 < /dev/null &
SERVER=$!

DISK=nvme0n1
sectors() { awk -v d=$DISK '$3 == d { print $6 }' /proc/diskstats; }
S0=$(sectors)
echo "utc_time,mem_available_mib,cached_mib,server_rss_mib,nvme_read_mib" > "$P.ram.csv"
while kill -0 "$SERVER" 2>/dev/null; do
  rss=$(ps -o rss= -g "$SERVER" 2>/dev/null | awk '{ s += $1 } END { print int(s / 1024) }')
  awk -v t="$(date -u +%T)" -v rss="$rss" -v rd="$(( ($(sectors) - S0) / 2048 ))" \
    '/^MemAvailable/ { a = int($2 / 1024) } /^Cached/ { c = int($2 / 1024) }
     END { print t "," a "," c "," rss "," rd }' /proc/meminfo >> "$P.ram.csv"
  sleep 1
done &
RAM_SAMPLER=$!

ready=0
for _ in $(seq 2400); do
  # Strata's /health answers 200 with "loaded": false while the model loads
  if h=$(curl -fs "$URL/health" 2>/dev/null) && ! grep -qE '"loaded": *false' <<< "$h"; then ready=1; break; fi
  kill -0 "$SERVER" 2>/dev/null || break
  sleep 0.5
done
load_s=$(awk -v a="$(date +%s.%N)" -v b="$t0" 'BEGIN { printf "%.1f", a - b }')
if [ $ready != 1 ]; then
  echo "!! FAILED: server not ready after ${load_s} s; last lines of $P.out:"
  tail -n 20 "$P.out"
  exit 1
fi
echo "== ready in ${load_s} s"
[ "$REPS" = 0 ] || python3 "$here/oai-bench.py" --url "$URL" --label "$LABEL" --out "$OUT/results.jsonl" --reps "$REPS" --load-s "$load_s"
echo "== peak VRAM per GPU (MiB):"
awk -F', ' 'NR > 1 { split($6, m, " "); if (m[1] > max[$2]) max[$2] = m[1] } END { for (g in max) print "  GPU" g ": " max[g] }' "$P.gpu.csv"
echo "== peak server RSS / lowest MemAvailable (MiB): $(awk -F, 'NR > 1 { if ($4 > r) r = $4; if (a == "" || $2 < a) a = $2 } END { print r " / " a }' "$P.ram.csv")"
