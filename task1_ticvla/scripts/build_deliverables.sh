#!/usr/bin/env bash
# IMS2026 HW0 Task 1 - assemble the visual deliverable from archived frames.
# Produces one side-by-side (RGB | third-person) mp4 per episode.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/yehzr/anaconda3/etc/profile.d/conda.sh
conda activate hw0-lerobot

OUT="${HERE}/outputs/videos"
mkdir -p "$OUT"
for d in "${HERE}"/outputs/episode_frames/episode_*; do
  [[ -d "$d" ]] || continue
  ep="$(basename "$d")"
  n=$(find "$d" -name '*.jpg' | wc -l)
  if [[ "$n" -eq 0 ]]; then
    echo "  ${ep}: no frames, skipped"
    continue
  fi
  python "${HERE}/scripts/make_videos.py" "$d" --out "$OUT" 2>&1 | grep -E "\.mp4" | sed "s/^/  ${ep} /"
done

echo
echo "=== deliverables ==="
ls -1sh "$OUT" | tail -n +2
