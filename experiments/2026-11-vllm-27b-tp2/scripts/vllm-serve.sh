#!/usr/bin/env bash
# Start vLLM in Docker from a run config, in the foreground, the way scripts/bench-run.sh expects a server.
#
#   scripts/vllm-serve.sh configs/<name>.args
#
# The .args file: the first non-comment line is the image (local/vllm-openai:v0.30.0 or v0.32.0), lines of the form
# ENV=VALUE are container environment variables, every other line is one vLLM argument (as in the source's compose
# file, one list item per line). The model directory is mounted read-only at /model, vLLM's cache at
# /root/.cache/vllm; the API is published on 127.0.0.1:8080 (the port the shared clients use).
# SIGTERM or SIGINT stops the container (docker stop), so killing this script never leaves a container running.
set -euo pipefail

cfg=${1:?config file}
MODEL_DIR=${MODEL_DIR:-$HOME/models/qwen3.8-27b-nvfp4-0cc2795}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/vllm-27b}
NAME=${VLLM_CONTAINER:-vllm-bench}
mkdir -p "$CACHE_DIR"

image= ; envs=() ; args=()
while IFS= read -r line || [[ -n $line ]]; do
  line=${line%%#*}; line=${line%"${line##*[![:space:]]}"}; line=${line#"${line%%[![:space:]]*}"}
  [[ -z $line ]] && continue
  if [[ -z $image ]]; then image=$line
  elif [[ $line =~ ^[A-Z_][A-Z0-9_]*= ]]; then envs+=(-e "$line")
  else args+=("$line"); fi
done < "$cfg"
[[ -n $image ]] || { echo "no image in $cfg" >&2; exit 1; }

docker rm -f "$NAME" > /dev/null 2>&1 || true
# -t 15: vLLM 0.30.0's shutdown can hang after it signals EngineCore; docker stop must end (SIGKILL) well inside
# bench-run.sh's 30 s before it kills this script's process group, docker stop with it, and leaves the container up.
stop() { docker stop -t 15 "$NAME" > /dev/null 2>&1 || true; }
trap 'stop; exit 0' TERM INT

echo "image: $image ($(docker image inspect "$image" --format '{{index .RepoDigests 0}}'))"
echo "model: $MODEL_DIR ($(cat "$MODEL_DIR/REVISION" 2>/dev/null))"
echo "args: ${args[*]}"
docker run --rm --name "$NAME" --gpus all --ipc host \
  -p 127.0.0.1:8080:8000 \
  -v "$MODEL_DIR":/model:ro -v "$CACHE_DIR":/root/.cache/vllm \
  "${envs[@]}" "$image" --model /model "${args[@]}" &
wait $!
