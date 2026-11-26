#!/usr/bin/env bash
# Download files from a Hugging Face model repository at a pinned revision and verify them.
#
#   scripts/hf-fetch.sh <repo> <revision> <dest-dir> <path-in-repo>...
#   e.g. scripts/hf-fetch.sh ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF ed59f92… \
#          ~/models/qwen3.8-flash-next-gsq-rco-iq3xxs IQ3_XXS/…-00001-of-00002.gguf …
#
# The expected size and hash of every file come from the Hub API at that revision before the
# download starts: the SHA-256 (LFS object id) for LFS files, the git blob SHA-1 for small files kept
# in git (configs, tokenizers). LFS files are checked with O_DIRECT reads, so the check reads the
# disk, not the page cache the download just filled; git files by their blob hash. Files land flat in
# <dest-dir> with REVISION (repo and commit) and SHA256SUMS (SHA-256 of every file, the form
# scripts/move-model.sh checks) next to them. Needs the hf CLI in ~/venvs/hf.
set -euo pipefail

repo=${1:?repo}; rev=${2:?revision}; dest=${3:?destination}; shift 3
(( $# )) || { echo "no files given" >&2; exit 1; }
[[ $rev =~ ^[0-9a-f]{40}$ ]] || { echo "revision must be a full 40-character commit hash" >&2; exit 1; }
hf=${HF_CLI:-$HOME/venvs/hf/bin/hf}

mkdir -p "$dest"
cd "$dest"
: > SHA256SUMS.expected
: > GITBLOBS.expected
for path in "$@"; do
  # paths-info returns the size and, for an LFS file, its sha256; a file kept in git has only its blob oid
  read -r size kind oid < <(curl -fsS -X POST "https://huggingface.co/api/models/$repo/paths-info/$rev" \
      --data-urlencode "paths=$path" | python3 -c '
import json, sys
f = json.load(sys.stdin)[0]
print(f["size"], "lfs" if f.get("lfs") else "git", f["lfs"]["oid"] if f.get("lfs") else f["oid"])')
  if [[ $kind == lfs ]]; then echo "$oid  $(basename "$path")" >> SHA256SUMS.expected
  else echo "$oid  $(basename "$path")" >> GITBLOBS.expected; fi
  echo "$(basename "$path"): $size bytes, $kind $([[ $kind == lfs ]] && echo sha256 || echo blob) $oid"
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
while read -r oid name; do
  got=$(python3 -c 'import hashlib, sys; d = open(sys.argv[1], "rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "$name")
  if [[ $got == "$oid" ]]; then echo "$name: OK (git blob)"; sha256sum "$name" >> SHA256SUMS.expected
  else echo "$name: FAILED (blob $got)"; fail=1; fi
done < GITBLOBS.expected
echo "verify: $(( $(date +%s) - t )) s"
(( fail == 0 )) || { echo "checksum mismatch; files kept for inspection, SHA256SUMS not written" >&2; exit 1; }
mv SHA256SUMS.expected SHA256SUMS
rm -f GITBLOBS.expected
ls -l
