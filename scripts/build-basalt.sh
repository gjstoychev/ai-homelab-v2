#!/usr/bin/env bash
# Build Basalt from a release tag with CUDA for the RTX 5060 Ti (compute capability 12.0), Qwen3.8-Flash-Next only.
#
#   scripts/build-basalt.sh [tag]        default: v0.5.7
#
# Runs as the user and installs nothing outside ~/basalt: source in <tag>/src, the build folder in <tag>/src/build,
# the programs in <tag>/src/bin (Basalt's own layout: `bin/basalt serve` runs the Python server from the folder above
# bin/), the server's Python environment in <tag>/src/.venv (requirements.txt, pinned), what was built in
# <tag>/build-info.txt, and ~/basalt/current -> <tag> after a successful build.
# docs/building.md: "The build fetches nothing." BASALT_MODELS=qwen4exp leaves the other models out of the build.
set -euo pipefail

tag=${1:-v0.5.7}
root=${BASALT_ROOT:-$HOME/basalt}
dir=$root/$tag
src=$dir/src
cuda=/usr/local/cuda-13.0
url=https://github.com/jesdga95/basalt.git

[[ -x $cuda/bin/nvcc ]] || { echo "missing $cuda/bin/nvcc: install the toolchain first (docs/stack.md)" >&2; exit 1; }
[[ ! -e $src/build ]] || { echo "$src/build exists; remove it to rebuild" >&2; exit 1; }

# a clone left from an earlier attempt is reused only if it is exactly the tag
if [[ -d $src/.git ]]; then
  [[ $(git -C "$src" describe --tags --exact-match 2>/dev/null) == "$tag" ]] ||
    { echo "$src is not at $tag; remove it" >&2; exit 1; }
else
  mkdir -p "$dir"
  git clone --quiet --filter=tree:0 --branch "$tag" "$url" "$src"
fi
[[ -z $(git -C "$src" status --porcelain) ]] || { echo "$src has local changes" >&2; exit 1; }

# The venv first: the configure step probes for a Python with numpy (.venv/bin/python is one of its candidates).
python3 -m venv "$src/.venv"
"$src/.venv/bin/pip" install --quiet --require-virtualenv -r "$src/requirements.txt"

# 120 = the RTX 5060 Ti, also Basalt's default and the only compute capability it accepts; named so the cache says
# it. CUDAToolkit_ROOT: cudart and cuBLAS from this toolkit. Half the cores: nvcc needs the RAM.
cmake -G Ninja -S "$src" -B "$src/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBASALT_MODELS=qwen4exp \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" \
  -DCUDAToolkit_ROOT="$cuda"
start=$(date +%s)
cmake --build "$src/build" -j "$(( $(nproc) / 2 ))"
build_s=$(( $(date +%s) - start ))

exe=$src/bin/basalt
# A failed probe is recorded in the file; it does not abort after a finished build.
set +e
{
  echo "== date";      date -u
  echo "== source";    git -C "$src" describe --tags; git -C "$src" rev-parse HEAD
  echo "== build";     echo "${build_s} s with -j $(( $(nproc) / 2 ))"
  echo "== toolchain"; "$cuda/bin/nvcc" --version | tail -n2; cmake --version | head -n1; ninja --version; gcc --version | head -n1
  echo "== packages";  dpkg-query -W -f '${Package} ${Version}\n' cuda-nvcc-13-0 libcublas-dev-13-0 nvidia-driver-580-server-open
  echo "== cmake";     grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER|CUDAToolkit_ROOT|BASALT_[A-Z0-9_]*|PEER_ADA|BUILD_TESTING|WERROR):' \
                         "$src/build/CMakeCache.txt"
  echo "== files";     ls -l "$src/bin"
  echo "== gpu code";  "$cuda/bin/cuobjdump" --list-elf "$exe" 2>/dev/null | grep -oE 'sm_[0-9]+[a-z]?' | sort | uniq -c
  echo "== libraries"; ldd "$exe" | grep -E 'cublas|cudart|libcuda\.|nccl' || true
  echo "== version";   strings "$exe" | grep -oE 'engine=[0-9]+\.[0-9]+\.[0-9]+' | sort -u   # no --version flag
  echo "== python";    "$src/.venv/bin/python" --version; "$src/.venv/bin/pip" freeze
} > "$dir/build-info.txt" 2>&1
set -e

ln -sfn "$tag" "$root/current"
cat "$dir/build-info.txt"
