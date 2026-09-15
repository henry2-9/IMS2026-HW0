#!/usr/bin/env bash
# IMS2026 HW0 Task 1 - snapshot each episode's camera frames.
#
# Every episode writes into the same logs/<run_id>/<robot>_ticvla_data/ directory
# and restarts its frame numbering at zero, so episode N+1 overwrites episode N.
# Watch the benchmark log and, each time a new episode starts, move the finished
# episode's frames into outputs/episode_frames/<episode>/.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${HERE}/outputs/bench_run.log"
DEST="${HERE}/outputs/episode_frames"
prev=""

current_episode() {
  # "Launching episode (subprocess): X" is printed by the parent as soon as the
  # previous episode finishes; the child's own "Running episode" line lags by the
  # ~3 minutes Isaac Sim takes to boot, which is too late to archive in time.
  grep -oP "Launching episode \(subprocess\): \K.*" "$LOG" 2>/dev/null | tail -1
}

snapshot() {
  local ep="$1"
  [[ -z "$ep" ]] && return
  while IFS= read -r d; do
    local out="${DEST}/${ep}/$(basename "$d")"
    mkdir -p "$out"
    # The container runs as root and its log directory is not writable by the
    # host user, so the frames must be copied, not moved.
    # Copy only the frames *this* episode wrote. Both robot directories persist
    # for the whole run and episodes restart frame numbering at zero, so without
    # a time filter an episode that produced few frames (a robot that never
    # moved, say) silently inherits the previous episode's footage and its video
    # ends up showing the wrong scene entirely.
    #
    # -newer against a marker file dropped when the episode started is exact,
    # unlike a fixed "-newermt -10 minutes" window that depends on how long the
    # episode happened to take.
    MARK="${DEST}/.${ep}.start"
    if [[ ! -f "$MARK" ]]; then
      cp -a "$d"/*.jpg "$out"/ 2>/dev/null
    else
      find "$d" -maxdepth 2 -name '*.jpg' -newer "$MARK" -exec cp -a -t "$out" {} + 2>/dev/null
    fi
    if [[ -z "$(find "$out" -name '*.jpg' -print -quit 2>/dev/null)" ]]; then
      echo "$(date +%H:%M:%S) ${ep}/$(basename "$d"): 無新影格,略過"
      rmdir "$out" 2>/dev/null
      continue
    fi
    local n; n=$(find "$out" -name '*.jpg' 2>/dev/null | wc -l)
    echo "$(date +%H:%M:%S) 歸檔 ${ep}/$(basename "$d"): ${n} 張"
  done < <(find "${HERE}/outputs/logs" -type d -name '*_ticvla_data' 2>/dev/null)
}

while true; do
  # stop once the benchmark container is gone and the last episode is archived
  if [[ -z "$(docker ps -q --filter ancestor=ims2026-hw0-task1:latest 2>/dev/null)" ]]; then
    snapshot "$(current_episode)"
    echo "容器結束,歸檔完成 $(date -Is)"
    break
  fi
  cur="$(current_episode)"
  if [[ -n "$cur" && "$cur" != "$prev" ]]; then
    [[ -n "$prev" ]] && snapshot "$prev"
    prev="$cur"
    mkdir -p "$DEST"; touch "${DEST}/.${cur}.start"   # marker for the time filter
    echo "$(date +%H:%M:%S) 目前執行: $cur"
  fi
  sleep 15
done
