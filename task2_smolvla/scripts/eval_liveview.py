#!/usr/bin/env python
"""IMS2026 HW0 - Task 2: LIBERO evaluation for SmolVLA with a live-view window.

Wraps LeRobot's own env / policy / processor stack (same construction as
`lerobot-eval`) but drives the rollout loop directly so we can render the
live-view window the assignment requires:

  * front camera (agentview) and wrist camera (eye-in-hand), side by side
  * task suite and task instruction
  * current episode index and step count / step limit
  * per-task and running total success statistics

Every episode is written to an mp4 so the full 400-episode run can be replayed.
Results are aggregated into eval_info.json.

Usage
-----
    # full protocol: 4 suites x 10 tasks x 10 episodes = 400 episodes
    python eval_liveview.py --policy-path <ckpt> --output-dir ../eval_logs/full

    # live demo: LIBERO-Long only, 1 episode per task
    python eval_liveview.py --policy-path <ckpt> --suites libero_10 \
        --n-episodes 1 --output-dir ../eval_logs/demo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# LIBERO suites, in the order the SmolVLA paper reports them.
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABEL = {
    "libero_spatial": "LIBERO-Spatial",
    "libero_object": "LIBERO-Object",
    "libero_goal": "LIBERO-Goal",
    "libero_10": "LIBERO-Long",
}
# Reported in the SmolVLA paper (arXiv:2506.01844), Table 2. The assignment
# requires each suite to land within +/-3 percentage points of these.
PAPER_SR = {
    "libero_spatial": 90.0,
    "libero_object": 96.0,
    "libero_goal": 92.0,
    "libero_10": 71.0,
}

logger = logging.getLogger("hw0-eval")


# --------------------------------------------------------------------------- #
# live-view rendering
# --------------------------------------------------------------------------- #
class LiveView:
    """Composes and shows the live-view window, and records it to disk."""

    PANEL_H = 150          # header strip height (px)
    CAM_SIZE = 360         # each camera pane is rendered at CAM_SIZE x CAM_SIZE

    def __init__(self, display: bool, record_dir: Path | None, fps: int = 20):
        self.display = display
        self.record_dir = record_dir
        self.fps = fps
        self._writer = None
        self._cv2 = None
        self._window = "IMS2026 HW0 | SmolVLA x LIBERO"
        if display:
            import cv2  # imported lazily so headless runs need no GUI libs

            self._cv2 = cv2
            cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        if record_dir is not None:
            record_dir.mkdir(parents=True, exist_ok=True)

    # -- video ------------------------------------------------------------- #
    def open_episode(self, path: Path) -> None:
        if self.record_dir is None:
            return
        import imageio.v2 as imageio

        self._writer = imageio.get_writer(str(path), fps=self.fps, macro_block_size=1)

    def close_episode(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    # -- drawing ----------------------------------------------------------- #
    @staticmethod
    def _upright(img: np.ndarray) -> np.ndarray:
        """robosuite returns frames flipped; match LiberoEnv.render()."""
        return img[::-1, ::-1]

    def _pane(self, img: np.ndarray, label: str) -> np.ndarray:
        cv2 = self._cv2 or __import__("cv2")
        img = self._upright(np.ascontiguousarray(img))
        img = cv2.resize(img, (self.CAM_SIZE, self.CAM_SIZE), interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img[:, :, ::-1])  # RGB -> BGR for cv2 drawing
        cv2.rectangle(img, (0, self.CAM_SIZE - 26), (self.CAM_SIZE, self.CAM_SIZE), (0, 0, 0), -1)
        cv2.putText(img, label, (8, self.CAM_SIZE - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def render(self, front, wrist, *, suite, instruction, task_idx, n_tasks,
               episode, n_episodes, step, max_steps, task_succ, task_done,
               total_succ, total_done):
        cv2 = self._cv2 or __import__("cv2")
        panes = np.hstack([self._pane(front, "front (agentview)"),
                           self._pane(wrist, "wrist (eye-in-hand)")])
        w = panes.shape[1]
        header = np.full((self.PANEL_H, w, 3), 24, dtype=np.uint8)

        def put(text, y, scale=0.6, color=(235, 235, 235), thick=1):
            cv2.putText(header, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thick, cv2.LINE_AA)

        put(f"{SUITE_LABEL.get(suite, suite)}   task {task_idx + 1}/{n_tasks}", 26, 0.66, (120, 220, 255), 2)

        # wrap the instruction over at most two lines
        instr = instruction or "(no instruction)"
        max_chars = max(20, int(w / 9.2))
        if len(instr) <= max_chars:
            put(f'"{instr}"', 50, 0.55, (255, 255, 255))
        else:
            cut = instr.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            tail = instr[cut:].strip()
            if len(tail) > max_chars:
                tail = tail[: max_chars - 1] + "\u2026"
            put(f'"{instr[:cut]}', 50, 0.55, (255, 255, 255))
            put(f'{tail}"', 70, 0.55, (255, 255, 255))
        y_stats = 96

        put(f"episode {episode + 1}/{n_episodes}    step {step}/{max_steps}", y_stats, 0.58, (200, 200, 200))

        task_sr = 100.0 * task_succ / task_done if task_done else 0.0
        tot_sr = 100.0 * total_succ / total_done if total_done else 0.0
        put(f"task SR {task_succ}/{task_done} ({task_sr:5.1f}%)     "
            f"total SR {total_succ}/{total_done} ({tot_sr:5.1f}%)",
            y_stats + 26, 0.58, (150, 255, 150))

        frame = np.vstack([header, panes])

        if self._writer is not None:
            self._writer.append_data(frame[:, :, ::-1])  # back to RGB for the file
        if self.display:
            cv2.imshow(self._window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt("live view closed by user")

    def close(self):
        self.close_episode()
        if self.display and self._cv2 is not None:
            self._cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cameras_from_obs(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pull (front, wrist) uint8 HWC frames out of a vector-env observation."""
    pixels = obs["pixels"]
    front = pixels["image"]
    wrist = pixels["image2"]
    # vector env adds a leading batch dim
    if front.ndim == 4:
        front, wrist = front[0], wrist[0]
    return np.asarray(front, dtype=np.uint8), np.asarray(wrist, dtype=np.uint8)


def _gpu_snapshot() -> dict:
    out = {"device": None, "driver": None, "peak_vram_gb": None}
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    try:
        out["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", required=True,
                   help="local checkpoint dir (…/pretrained_model) or Hub id")
    p.add_argument("--suites", default=",".join(SUITE_ORDER),
                   help="comma-separated LIBERO suites")
    p.add_argument("--task-ids", default=None,
                   help="comma-separated task indices; default = all 10 per suite")
    p.add_argument("--n-episodes", type=int, default=10, help="episodes per task")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="override policy.n_action_steps (chunk executed per inference)")
    p.add_argument("--no-display", action="store_true",
                   help="skip the on-screen window (still records mp4)")
    p.add_argument("--no-record", action="store_true", help="skip mp4 recording")
    p.add_argument("--control-mode", default="relative", choices=["relative", "absolute"])
    p.add_argument("--flip-obs", action="store_true",
                   help="rotate observations 180 deg before the policy sees them. "
                        "robosuite renders bottom-up, so LiberoEnv._format_raw_obs "
                        "hands the policy an inverted image while the lerobot/libero "
                        "training frames are upright — LiberoEnv.render() applies "
                        "exactly this correction but only for display.")
    p.add_argument("--obs-size", type=int, default=256,
                   help="camera render size fed to the policy. LeRobot's LiberoEnv "
                        "defaults to 360, but the lerobot/libero dataset (and hence "
                        "the trained checkpoint) is 256x256 - leaving them mismatched "
                        "is a train/eval distribution shift.")
    p.add_argument("--rename-map", default=None,
                   help='JSON dict remapping env observation keys to the names the '
                        'checkpoint expects, e.g. \'{"observation.images.image": '
                        '"observation.images.camera1"}\'. Needed only for checkpoints '
                        'trained on a differently-named camera schema.')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Imported after env vars are set.
    from libero.libero import benchmark
    from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION
    from lerobot.utils.random_utils import set_seed

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = None if args.no_record else out_dir / "videos"

    rename_map = json.loads(args.rename_map) if args.rename_map else {}

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    task_ids = ([int(t) for t in args.task_ids.split(",")] if args.task_ids else None)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    view = LiveView(display=not args.no_display, record_dir=video_dir)

    wall_start = time.time()
    results: dict[str, dict] = {}
    per_task_records: list[dict] = []
    policy = None
    total_succ = total_done = 0

    try:
        for suite_name in suites:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            ids = list(range(len(suite.tasks))) if task_ids is None else task_ids
            suite_succ = suite_done = 0

            for t_pos, task_id in enumerate(ids):
                env_cfg = LiberoEnvConfig(
                    task=suite_name,
                    task_ids=[task_id],
                    control_mode=args.control_mode,
                    init_states=True,
                    hard_reset=True,   # bit-identical to the published protocol
                    observation_height=args.obs_size,
                    observation_width=args.obs_size,
                )
                envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
                env = envs[suite_name] if isinstance(envs, dict) else envs
                if isinstance(env, dict):
                    env = next(iter(env.values()))

                if policy is None:
                    policy = make_policy(cfg=_policy_cfg(args, env_cfg), env_cfg=env_cfg,
                                         rename_map=rename_map)
                    if args.n_action_steps is not None:
                        policy.config.n_action_steps = args.n_action_steps
                    policy.eval()
                    pre, post = make_pre_post_processors(
                        policy_cfg=policy.config,
                        pretrained_path=args.policy_path,
                        preprocessor_overrides={
                            "device_processor": {"device": str(policy.config.device)},
                            "rename_observations_processor": {"rename_map": rename_map},
                        },
                    )
                    env_pre, env_post = make_env_pre_post_processors(
                        env_cfg=env_cfg, policy_cfg=policy.config)
                    use_amp = bool(getattr(policy.config, "use_amp", False))
                    device = torch.device(policy.config.device)

                instruction = suite.tasks[task_id].language
                max_steps = env.call("_max_episode_steps")[0]
                task_succ = task_done = 0

                for ep in range(args.n_episodes):
                    seed = args.seed + 1000 * task_id + ep
                    policy.reset()
                    obs, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})

                    if video_dir is not None:
                        view.open_episode(
                            video_dir / f"{suite_name}_task{task_id:02d}_ep{ep:02d}.mp4")

                    success = False
                    step = 0
                    amp_ctx = (torch.autocast(device_type=device.type) if use_amp
                               else nullcontext())
                    with torch.no_grad(), amp_ctx:
                        while step < max_steps:
                            front, wrist = _cameras_from_obs(obs)
                            view.render(
                                front, wrist,
                                suite=suite_name, instruction=instruction,
                                task_idx=t_pos, n_tasks=len(ids),
                                episode=ep, n_episodes=args.n_episodes,
                                step=step, max_steps=max_steps,
                                task_succ=task_succ, task_done=task_done,
                                total_succ=total_succ, total_done=total_done,
                            )

                            if args.flip_obs:
                                obs = dict(obs)
                                # ascontiguousarray: the reversed views have
                                # negative strides, which torch.from_numpy rejects.
                                obs["pixels"] = {
                                    k: np.ascontiguousarray(
                                        v[..., ::-1, ::-1, :] if v.ndim == 4 else v[::-1, ::-1]
                                    )
                                    for k, v in obs["pixels"].items()
                                }
                            processed = preprocess_observation(obs)
                            try:
                                processed["task"] = list(env.call("task_description"))
                            except (AttributeError, NotImplementedError):
                                processed["task"] = [instruction]
                            processed = env_pre(processed)
                            processed = pre(processed)
                            with torch.inference_mode():
                                action = policy.select_action(processed)
                            action = post(action)
                            action = env_post({ACTION: action})[ACTION]
                            action_np = action.to("cpu").numpy()

                            obs, _reward, terminated, truncated, info = env.step(action_np)
                            step += 1

                            success = success or _extract_success(info)
                            if success or bool(np.any(terminated)) or bool(np.any(truncated)):
                                break

                    view.close_episode()
                    task_done += 1
                    total_done += 1
                    suite_done += 1
                    if success:
                        task_succ += 1
                        total_succ += 1
                        suite_succ += 1

                    logger.info(
                        "%s task %d ep %d/%d -> %s | task %d/%d | total %d/%d (%.1f%%)",
                        SUITE_LABEL.get(suite_name, suite_name), task_id, ep + 1,
                        args.n_episodes, "SUCCESS" if success else "fail",
                        task_succ, task_done, total_succ, total_done,
                        100.0 * total_succ / max(1, total_done))

                per_task_records.append({
                    "suite": suite_name,
                    "task_id": task_id,
                    "instruction": instruction,
                    "n_episodes": task_done,
                    "successes": task_succ,
                    "success_rate": 100.0 * task_succ / max(1, task_done),
                })
                env.close()

            sr = 100.0 * suite_succ / max(1, suite_done)
            results[suite_name] = {
                "label": SUITE_LABEL.get(suite_name, suite_name),
                "n_episodes": suite_done,
                "successes": suite_succ,
                "success_rate": sr,
                "paper_success_rate": PAPER_SR.get(suite_name),
                "delta_vs_paper": (None if suite_name not in PAPER_SR
                                   else round(sr - PAPER_SR[suite_name], 2)),
                "within_3pp": (None if suite_name not in PAPER_SR
                               else abs(sr - PAPER_SR[suite_name]) <= 3.0),
            }
            logger.info("== %s: %.1f%% (paper %.1f%%) ==",
                        SUITE_LABEL.get(suite_name, suite_name), sr,
                        PAPER_SR.get(suite_name, float("nan")))
    except KeyboardInterrupt:
        logger.warning("interrupted - writing partial results")
    finally:
        view.close()

    elapsed = time.time() - wall_start
    srs = [v["success_rate"] for v in results.values()]
    eval_info = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_path": args.policy_path,
        "seed": args.seed,
        "control_mode": args.control_mode,
        "n_action_steps": args.n_action_steps,
        "episodes_per_task": args.n_episodes,
        "per_suite": results,
        "per_task": per_task_records,
        "aggregate": {
            "total_episodes": total_done,
            "total_successes": total_succ,
            "overall_success_rate": 100.0 * total_succ / max(1, total_done),
            "mean_of_suite_success_rates": (float(np.mean(srs)) if srs else None),
            "paper_average": 87.3,
        },
        "compute": {
            **_gpu_snapshot(),
            "wall_clock_s": round(elapsed, 1),
            "wall_clock_h": round(elapsed / 3600.0, 3),
            "s_per_episode": round(elapsed / max(1, total_done), 2),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    path = out_dir / "eval_info.json"
    path.write_text(json.dumps(eval_info, indent=2))
    logger.info("wrote %s", path)

    print("\n" + "=" * 68)
    print(f"{'suite':<18}{'ours':>10}{'paper':>10}{'delta':>10}{'within 3pp':>14}")
    print("-" * 68)
    for s in SUITE_ORDER:
        if s not in results:
            continue
        r = results[s]
        print(f"{r['label']:<18}{r['success_rate']:>9.1f}%{r['paper_success_rate']:>9.1f}%"
              f"{r['delta_vs_paper']:>+10.1f}{('YES' if r['within_3pp'] else 'NO'):>14}")
    print("-" * 68)
    if srs:
        print(f"{'average':<18}{float(np.mean(srs)):>9.1f}%{87.3:>9.1f}%")
    print(f"{total_done} episodes in {elapsed/3600:.2f} h")
    print("=" * 68)
    return 0


def _extract_success(info) -> bool:
    """Read is_success out of a (possibly vectorised) gym info dict."""
    if isinstance(info, dict):
        if "final_info" in info:
            fi = info["final_info"]
            if isinstance(fi, dict) and "is_success" in fi:
                return bool(np.any(fi["is_success"]))
            if isinstance(fi, (list, tuple, np.ndarray)):
                return any(bool(x.get("is_success", False))
                           for x in fi if isinstance(x, dict))
        if "is_success" in info:
            return bool(np.any(info["is_success"]))
    return False


def _policy_cfg(args, env_cfg):
    """Build the PreTrainedConfig for the checkpoint under evaluation."""
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = args.policy_path
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


if __name__ == "__main__":
    raise SystemExit(main())
