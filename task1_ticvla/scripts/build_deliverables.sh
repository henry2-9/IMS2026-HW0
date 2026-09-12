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

# Guarantee H.264: OpenCV may silently fall back to mpeg4, which Firefox cannot
# decode. Re-encode anything that is not already avc1, and add +faststart so the
# files stream without downloading in full.
for v in "$OUT"/*.mp4; do
  [[ -f "$v" ]] || continue
  codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$v")
  if [[ "$codec" != "h264" ]]; then
    ffmpeg -y -v error -i "$v" -c:v libx264 -preset medium -crf 23 \
      -pix_fmt yuv420p -movflags +faststart "${v%.mp4}.h264.mp4" </dev/null \
      && mv "${v%.mp4}.h264.mp4" "$v" && echo "  re-encoded $(basename "$v") ${codec} -> h264"
  fi
done

echo
echo "=== deliverables ==="
ls -1sh "$OUT" | tail -n +2
