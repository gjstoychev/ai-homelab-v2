#!/usr/bin/env bash
# Move a model directory between ~/models (NVMe) and /mnt/models (SATA archive).
#
#   scripts/move-model.sh <model-dir> <target-parent>
#   e.g. scripts/move-model.sh ~/models/deepseek-v4-flash-0731-ud-iq1m /mnt/models
#
# The directory must hold SHA256SUMS. The copy is checked against it with O_DIRECT reads,
# so the check reads what reached the target disk, not the page cache that the copy just
# filled. The source is deleted only after every file passes.
set -euo pipefail

src=$(realpath "${1:?model directory}")
dst=$(realpath "${2:?target parent directory}")/$(basename "$src")

[[ -f $src/SHA256SUMS ]] || { echo "no $src/SHA256SUMS" >&2; exit 1; }
if pids=$(fuser "$src"/* 2>/dev/null) && [[ -n $pids ]]; then
  echo "files in $src are open by: $pids" >&2; exit 1
fi
need=$(du -sb --exclude=.claude "$src" | cut -f1)
free=$(df -B1 --output=avail "$(dirname "$dst")" | tail -n1)
(( need < free )) || { echo "need $need bytes, $free free on $(dirname "$dst")" >&2; exit 1; }

t=$(date +%s)
# .claude/: empty placeholders left by the v1 Claude Code sandbox
rsync -a --exclude=.claude "$src/" "$dst/"
sync -f "$dst"
echo "copy: $(( $(date +%s) - t )) s, $need bytes"

t=$(date +%s)
cd "$dst"
fail=0
while read -r sum name; do
  got=$(dd if="$name" iflag=direct bs=16M status=none | sha256sum | cut -d' ' -f1)
  if [[ $got == "$sum" ]]; then echo "$name: OK"; else echo "$name: FAILED ($got)"; fail=1; fi
done < SHA256SUMS
echo "verify: $(( $(date +%s) - t )) s"
(( fail == 0 )) || { echo "checksum mismatch: $src kept" >&2; exit 1; }

rm -r "$src"
echo "moved to $dst; $src removed"
