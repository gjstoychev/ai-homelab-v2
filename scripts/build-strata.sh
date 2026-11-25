#!/usr/bin/env bash
# Build the Strata engine from a release tag with CUDA for the RTX 5060 Ti (compute capability 12.0).
#
#   scripts/build-strata.sh [tag]        default: v0.2.21
#
# Runs as the user and installs nothing outside ~/strata: source in <tag>/src, build in <tag>/build, the Python
# environment of the server and the pack tools in <tag>/src/terminal/.venv (terminal/requirements.txt, pinned),
# what was built in <tag>/build-info.txt, and ~/strata/current -> <tag> after a successful build.
# Strata's setup.sh is not used: it can install packages through apt. The CMake definitions are the ones its
# terminal/setup.py (build_engine) passes for an NVIDIA card with CUDA 13, without the image encoder.
set -euo pipefail

tag=${1:-v0.2.21}
root=${STRATA_ROOT:-$HOME/strata}
dir=$root/$tag
cuda=/usr/local/cuda-13.0
url=https://github.com/Niko1221/Strata.git

[[ -x $cuda/bin/nvcc ]] || { echo "missing $cuda/bin/nvcc: install the toolchain first (docs/stack.md)" >&2; exit 1; }
[[ ! -e $dir/build ]] || { echo "$dir/build exists; remove it to rebuild" >&2; exit 1; }

# a clone left from an earlier attempt is reused only if it is exactly the tag
if [[ -d $dir/src/.git ]]; then
  [[ $(git -C "$dir/src" describe --tags --exact-match 2>/dev/null) == "$tag" ]] ||
    { echo "$dir/src is not at $tag; remove it" >&2; exit 1; }
else
  mkdir -p "$dir"
  git clone --quiet --filter=tree:0 --branch "$tag" "$url" "$dir/src"
fi
[[ -z $(git -C "$dir/src" status --porcelain) ]] || { echo "$dir/src has local changes" >&2; exit 1; }

# As setup.py: Ninja, Release, tests off (they would fetch Catch2), the Q6_K expert kernels, the toolkit's own
# cudart and cuBLAS (CUDAToolkit_ROOT), not portable (built for this CPU), half the cores (nvcc needs the RAM).
# 120 = the RTX 5060 Ti; also the default ("RTX 50 only"), named so build-info.txt and the cache say it.
cmake -G Ninja -S "$dir/src/engine" -B "$dir/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CCACHE=OFF \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_Q6K_EXPERTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" \
  -DCUDAToolkit_ROOT="$cuda"
start=$(date +%s)
cmake --build "$dir/build" --target strata -j "$(( $(nproc) / 2 ))"
build_s=$(( $(date +%s) - start ))

venv=$dir/src/terminal/.venv
python3 -m venv "$venv"
"$venv/bin/pip" install --quiet --require-virtualenv -r "$dir/src/terminal/requirements.txt"

exe=$dir/build/strata
# A failed probe is recorded in the file; it does not abort after a finished build.
set +e
{
  echo "== date";      date -u
  echo "== source";    git -C "$dir/src" describe --tags; git -C "$dir/src" rev-parse HEAD
                       grep -m1 -E '^project\(strata VERSION' "$dir/src/engine/CMakeLists.txt"
  echo "== build";     echo "${build_s} s with -j $(( $(nproc) / 2 ))"
  echo "== toolchain"; "$cuda/bin/nvcc" --version | tail -n2; cmake --version | head -n1; ninja --version; gcc --version | head -n1
  echo "== packages";  dpkg-query -W -f '${Package} ${Version}\n' cuda-nvcc-13-0 libcublas-dev-13-0 nvidia-driver-580-server-open
  echo "== cmake";     grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER|CUDAToolkit_ROOT|STRATA_[A-Z0-9_]*|GGML_[A-Z_]*):' \
                         "$dir/build/CMakeCache.txt"
  echo "== files";     ls -l "$exe" "$dir/build"/libstrata_kq.so 2>&1
  echo "== gpu code";  "$cuda/bin/cuobjdump" --list-elf "$exe" 2>/dev/null | grep -oE 'sm_[0-9]+[a-z]?' | sort | uniq -c
                       for so in "$dir/build"/*.so; do [[ -e $so ]] && { echo "$(basename "$so"):"; "$cuda/bin/cuobjdump" --list-elf "$so" | grep -oE 'sm_[0-9]+[a-z]?' | sort | uniq -c; }; done
  echo "== libraries"; ldd "$exe" | grep -E 'cublas|cudart|libcuda\.|strata|nccl' || true
  echo "== version";   "$exe" --version 2>&1
  echo "== python";    "$venv/bin/python" --version; "$venv/bin/pip" freeze
} > "$dir/build-info.txt" 2>&1
set -e

ln -sfn "$tag" "$root/current"
cat "$dir/build-info.txt"
