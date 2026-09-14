#!/usr/bin/env python
"""IMS2026 HW0 Task 1 - live camera view for the DynaNav benchmark.

The benchmark runs headless and writes its two camera streams to disk as they
are produced:

    front_frame_NNNNNN.jpg / head_frame_NNNNNN.jpg   robot RGB (Carter / Spot)
    tp_frame_NNNNNN.jpg                              third-person

This tails those directories and shows the newest pair side by side, so the
robot's movement is visible during a live demo instead of only the terminal log.
It only reads files — the benchmark is untouched, and evaluation results are
unaffected.

    python live_view.py                    # watch outputs/logs*, newest run
    python live_view.py --dir <logs dir>   # watch one specific directory

Press q to close.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

PANE_W, PANE_H = 640, 360
LABEL_H = 30
HEADER_H = 46
RGB_PREFIXES = ("front_frame_", "head_frame_")
FRAME_RE = re.compile(r"_(\d{6})\.jpg$")


def _newest_jpg_mtime(d: Path) -> float:
    """mtime of the most recent jpg in this directory, or 0.0 if it has none."""
    best = 0.0
    for sub in (d, d / "rgb_keep"):
        if not sub.is_dir():
            continue
        try:
            for f in sub.iterdir():
                if f.name.endswith(".jpg"):
                    m = f.stat().st_mtime
                    if m > best:
                        best = m
        except OSError:
            continue
    return best


def newest_data_dir(roots: list[Path], max_age: float = 30.0) -> Path | None:
    """The *_ticvla_data directory that is actively being written to.

    Ranking by directory mtime is not enough: a finished run leaves its
    directory behind, and on a re-run the stale one can still look newer than a
    live one that has not written its first frame yet. Rank by the newest frame
    *file* instead, and ignore directories whose last frame is older than
    `max_age` seconds so the viewer does not latch onto a previous episode.
    """
    cands = [d for r in roots if r.is_dir()
             for d in r.rglob("*_ticvla_data") if d.is_dir()]
    if not cands:
        return None
    scored = [(d, _newest_jpg_mtime(d)) for d in cands]
    scored = [(d, m) for d, m in scored if m > 0.0]
    if not scored:
        return None
    d, m = max(scored, key=lambda x: x[1])
    if time.time() - m > max_age:
        return None
    return d


def latest_frame(d: Path, prefixes: tuple[str, ...]) -> tuple[int, Path] | None:
    """Highest-numbered frame with any of these prefixes, incl. the rgb_keep archive."""
    best: tuple[int, Path] | None = None
    for sub in (d, d / "rgb_keep"):
        if not sub.is_dir():
            continue
        try:
            entries = list(sub.iterdir())
        except OSError:
            continue
        for f in entries:
            if not f.name.endswith(".jpg") or not f.name.startswith(prefixes):
                continue
            m = FRAME_RE.search(f.name)
            if m and (best is None or int(m.group(1)) > best[0]):
                best = (int(m.group(1)), f)
    return best


def pane(path: Path | None, label: str) -> np.ndarray:
    img = None
    if path is not None:
        # The writer may still be flushing this file; a failed read is normal.
        img = cv2.imread(str(path))
    if img is None:
        img = np.full((PANE_H, PANE_W, 3), 28, np.uint8)
        cv2.putText(img, "waiting for frames...", (18, PANE_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1, cv2.LINE_AA)
    else:
        img = cv2.resize(img, (PANE_W, PANE_H), interpolation=cv2.INTER_AREA)
    strip = np.full((LABEL_H, PANE_W, 3), 18, np.uint8)
    cv2.putText(strip, label, (10, LABEL_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([img, strip])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=None,
                    help="a specific *_ticvla_data directory to watch")
    ap.add_argument("--root", type=Path, default=None,
                    help="outputs/ directory to search (default: ../outputs)")
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args()

    here = Path(__file__).resolve().parent.parent
    roots = [a.root] if a.root else sorted(here.glob("outputs/logs*"))
    win = "IMS2026 HW0 Task 1 - TIC-VLA live view"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    delay = max(1, int(1000 / a.fps))
    watched: Path | None = a.dir
    last_scan = 0.0

    try:
        while True:
            now = time.time()
            # Re-scan periodically so the window latches onto the next episode
            # by itself, and picks up a run that starts after the viewer.
            if a.dir is None and now - last_scan > 2.0:
                found = newest_data_dir(roots if roots else [here / "outputs"])
                if found is not None and found != watched:
                    watched = found
                last_scan = now

            rgb = tp = None
            step = None
            if watched is not None:
                r = latest_frame(watched, RGB_PREFIXES)
                t = latest_frame(watched, ("tp_frame_",))
                rgb = r[1] if r else None
                tp = t[1] if t else None
                step = max([x[0] for x in (r, t) if x], default=None)

            robot = "Spot" if watched and "spot" in watched.name else "Nova Carter"
            panes = np.hstack([pane(rgb, f"{robot}  |  RGB (robot camera)"),
                               pane(tp, f"{robot}  |  third-person view")])
            header = np.full((HEADER_H, panes.shape[1], 3), 22, np.uint8)
            ep = watched.parent.name if watched else "waiting for a run..."
            cv2.putText(header, f"{ep}   {robot}" + (f"   frame {step}" if step else ""),
                        (12, HEADER_H - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (120, 220, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, np.vstack([header, panes]))

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
