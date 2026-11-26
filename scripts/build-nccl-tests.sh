#!/usr/bin/env bash
# Build NVIDIA's nccl-tests from a release tag against the installed CUDA 13.0 toolkit and NCCL (docs/stack.md),
# for sm_120 only, without MPI (one process drives both GPUs: all_reduce_perf -g 2).
#
#   scripts/build-nccl-tests.sh [tag]        default: v2.22.0
#
# Installs nothing outside ~/nccl-tests: source in <tag>/src, the programs in <tag>/src/build, what was built in
# <tag>/build-info.txt, and ~/nccl-tests/current -> <tag> after a successful build.
set -euo pipefail

tag=${1:-v2.22.0}
dir=$HOME/nccl-tests/$tag
src=$dir/src
cuda=/usr/local/cuda-13.0
url=https://github.com/NVIDIA/nccl-tests.git

[[ -x $cuda/bin/nvcc ]] || { echo "missing $cuda/bin/nvcc: install the toolchain first (docs/stack.md)" >&2; exit 1; }
[[ ! -e $src/build ]] || { echo "$src/build exists; remove it to rebuild" >&2; exit 1; }

if [[ -d $src/.git ]]; then
  [[ $(git -C "$src" describe --tags --exact-match 2>/dev/null) == "$tag" ]] ||
    { echo "$src is not at $tag; remove it" >&2; exit 1; }
else
  mkdir -p "$dir"
  git clone --quiet --branch "$tag" --depth 1 "$url" "$src"
fi

make -C "$src" -j"$(nproc)" CUDA_HOME=$cuda NCCL_HOME=/usr NVCC_GENCODE="-gencode=arch=compute_120,code=sm_120" MPI=0
{
  echo "nccl-tests $tag, commit $(git -C "$src" rev-parse --short HEAD), built $(date -u +%FT%TZ)"
  echo "nvcc: $($cuda/bin/nvcc --version | tail -1)"
  echo "nccl: $(dpkg-query -W -f='${Version}' libnccl2)"
} > "$dir/build-info.txt"
ln -sfn "$dir" "$HOME/nccl-tests/current"
cat "$dir/build-info.txt"
