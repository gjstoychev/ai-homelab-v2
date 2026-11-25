#!/usr/bin/env bash
# Download files from a Hugging Face model repository at a pinned revision and verify them.
#
#   scripts/hf-fetch.sh <repo> <revision> <dest-dir> <path-in-repo>...
#   e.g. scripts/hf-fetch.sh ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF ed59f92… \
#          ~/models/qwen3.8-flash-next-gsq-rco-iq3xxs IQ3_XXS/…-00001-of-00002.gguf …
#
# The expected size and SHA-256 of every file come from the Hub API at that revision (the LFS
# object id) before the download starts. Files land flat in <dest-dir> with REVISION (repo and
# commit) and SHA256SUMS next to them; the check reads with O_DIRECT, so it reads the disk,
# not the page cache the download just filled. Needs the hf CLI in ~/venvs/hf.
set -euo pipefail

repo=${1:?repo}; rev=${2:?revision}; dest=${3:?destination}; shift 3
(( $# )) || { echo "no files given" >&2; exit 1; }
[[ $rev =~ ^[0-9a-f]{40}$ ]] || { echo "revision must be a full 40-character commit hash" >&2; exit 1; }
hf=${HF_CLI:-$HOME/venvs/hf/bin/hf}

mkdir -p "$dest"
cd "$dest"
: > SHA256SUMS.expected
for path in "$@"; do
  # paths-info returns size and LFS sha256 of the file at exactly this revision
  read -r size sha < <(curl -fsS -X POST "https://huggingface.co/api/models/$repo/paths-info/$rev" \
      --data-urlencode "paths=$path" | python3 -c '
import json, sys
f = json.load(sys.stdin)[0]
print(f["size"], f["lfs"]["oid"])')
  echo "$sha  $(basename "$path")" >> SHA256SUMS.expected
  echo "$(basename "$path"): $size bytes, sha256 $sha"
done
echo "$repo $rev" > REVISION

t=$(date +%s)
"$hf" download "$repo" "$@" --revision "$rev" --local-dir .
for path in "$@"; do
  [[ $path == */* ]] && mv "$path" .
done
rm -rf .cache
find . -mindepth 1 -type d -empty -delete
echo "download: $(( $(date +%s) - t )) s"

t=$(date +%s)
fail=0
while read -r sum name; do
  got=$(dd if="$name" iflag=direct bs=16M status=none | sha256sum | cut -d' ' -f1)
  if [[ $got == "$sum" ]]; then echo "$name: OK"; else echo "$name: FAILED ($got)"; fail=1; fi
done < SHA256SUMS.expected
echo "verify: $(( $(date +%s) - t )) s"
(( fail == 0 )) || { echo "checksum mismatch; files kept for inspection, SHA256SUMS not written" >&2; exit 1; }
mv SHA256SUMS.expected SHA256SUMS
ls -l
