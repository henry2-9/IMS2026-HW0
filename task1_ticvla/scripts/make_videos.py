#!/usr/bin/env python
"""IMS2026 HW0 Task 1 - build the side-by-side visual deliverable.

The assignment asks for a GIF/video/image sequence covering two robot platforms
across at least four scenes, showing the RGB view and a third-person view.

The TIC-VLA behaviour scripts write those two streams as separate JPEG series
under DynaNav/logs/<run_id>/<robot>_ticvla_data/:

    front_frame_NNNNNN.jpg / head_frame_NNNNNN.jpg   robot camera (Carter / Spot)
    tp_frame_NNNNNN.jpg                              third-person camera

The robot-camera frames double as the model's input buffer and are rotated into
`rgb_keep/` by our patch, so both directories are searched.

    python make_videos.py <logs-dir> --out <videos-dir>
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PANE_W, PANE_H = 640, 360
LABEL_H = 28
FRAME_RE = re.compile(r"_(\d{6})\.jpg$")


def index_frames(d: Path, prefixes: tuple[str, ...]) -> dict[int, Path]:
    """Map frame number -> path, searching the directory and its rgb_keep/."""
    out: dict[int, Path] = {}
    for sub in (d, d / "rgb_keep"):
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if not f.name.endswith(".jpg") or not f.name.startswith(prefixes):
                continue
            m = FRAME_RE.search(f.name)
            if m:
                out.setdefault(int(m.group(1)), f)
    return out


def pane(path: Path | None, label: str) -> np.ndarray:
    img = cv2.imread(str(path)) if path else None
    if img is None:
        img = np.full((PANE_H, PANE_W, 3), 32, np.uint8)
        cv2.putText(img, "no frame", (16, PANE_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 110), 1, cv2.LINE_AA)
    else:
        img = cv2.resize(img, (PANE_W, PANE_H), interpolation=cv2.INTER_AREA)
    strip = np.full((LABEL_H, PANE_W, 3), 20, np.uint8)
    cv2.putText(strip, label, (10, LABEL_H - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def build(data_dir: Path, out_path: Path, fps: int) -> int:
    robot = "Spot" if "spot" in data_dir.name else "Nova Carter"
    rgb = index_frames(data_dir, ("front_frame_", "head_frame_"))
    tp = index_frames(data_dir, ("tp_frame_",))
    if not rgb and not tp:
        return 0

    # The two cameras are sampled on the same cadence but a frame can be missing
    # from either stream; pair each third-person frame with the nearest earlier
    # robot-camera frame so the views stay in sync.
    rgb_keys = sorted(rgb)
    frames = sorted(tp) or rgb_keys
    writer = None
    n = 0
    for k in frames:
        prior = [x for x in rgb_keys if x <= k]
        left = rgb.get(prior[-1]) if prior else (rgb[rgb_keys[0]] if rgb_keys else None)
        canvas = np.hstack([pane(left, f"{robot}  |  RGB (robot camera)"),
                            pane(tp.get(k), f"{robot}  |  third-person view")])
        if writer is None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # OpenCV's "mp4v" is MPEG-4 Part 2, which Chrome plays but Firefox
            # does not. Prefer H.264 ("avc1"); fall back to mp4v and re-encode
            # afterwards if this build of OpenCV has no H.264 encoder.
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"),
                                     fps, (canvas.shape[1], canvas.shape[0]))
            if not writer.isOpened():
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
        n += 1
    if writer is not None:
        writer.release()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs_dir", type=Path, help="outputs/logs (or a single run_id dir)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    dirs = sorted(p for p in a.logs_dir.rglob("*_ticvla_data") if p.is_dir())
    if not dirs:
        print(f"no *_ticvla_data directories under {a.logs_dir}")
        return 1

    total = 0
    for d in dirs:
        # <run_id>/<robot>_ticvla_data -> <run_id>_<robot>.mp4
        name = f"{d.parent.name}_{d.name.replace('_ticvla_data','')}.mp4"
        n = build(d, a.out / name, a.fps)
        print(f"  {name:<44} {n:>5} frames")
        total += n
    print(f"\n{len(dirs)} clip(s), {total} frames -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
