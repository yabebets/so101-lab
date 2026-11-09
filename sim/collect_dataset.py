"""Record scripted-expert pick-and-place episodes into a LeRobot (v3) dataset.

Usage:  .venv/bin/python sim/collect_dataset.py --episodes 50 --root sim/data/so101_sim_pick_place
Each frame: observation.images.front (HxWx3), observation.state (6 joint positions, rad),
action (6 joint targets, rad). Only successful episodes are kept.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from so101_env import JOINTS, SO101PickPlaceEnv  # noqa: E402
from so101_expert import plan  # noqa: E402

from lerobot.datasets.feature_utils import hw_to_dataset_features  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

TASK = "Pick up the white block and place it in the tray."

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--root", type=Path, default=Path(__file__).parent / "data" / "so101_sim_pick_place")
ap.add_argument("--repo-id", default="local/so101_sim_pick_place")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--height", type=int, default=240)
ap.add_argument("--width", type=int, default=320)
ap.add_argument("--seed0", type=int, default=1000, help="first env seed (eval uses seeds < 1000)")
ap.add_argument("--action-noise", type=float, default=0.0, help="DART-style: std (rad) of noise added to the *executed* arm action; the clean expert action is recorded")
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

if args.root.exists():
    if not args.overwrite:
        sys.exit(f"{args.root} exists; pass --overwrite to replace it")
    shutil.rmtree(args.root)

motor_fts = {f"{j}.pos": float for j in JOINTS}
features = {
    **hw_to_dataset_features({**motor_fts, "front": (args.height, args.width, 3)}, "observation", use_video=True),
    **hw_to_dataset_features(motor_fts, "action", use_video=True),
}
ds = LeRobotDataset.create(args.repo_id, args.fps, features, root=args.root, robot_type="so101_sim", use_videos=True)

env = SO101PickPlaceEnv(fps=args.fps, image_hw=(args.height, args.width))
rng = np.random.default_rng(0)
kept, seed, t0 = 0, args.seed0, time.time()
while kept < args.episodes:
    obs = env.reset(seed)
    seed += 1
    try:
        traj = plan(env, env.block_pos)
    except RuntimeError as e:
        print(f"seed {seed - 1}: skipped ({e})")
        continue
    frames, success = [], False
    for q in traj:
        a = q.copy()
        if args.action_noise > 0:
            a[:5] += rng.normal(0, args.action_noise, 5)
        frames.append({"observation.images.front": obs["image"], "observation.state": obs["state"].astype(np.float32), "action": q.astype(np.float32), "task": TASK})
        obs, success = env.step(a)
    if not success:
        print(f"seed {seed - 1}: expert failed, skipped")
        continue
    for f in frames:
        ds.add_frame(f)
    ds.save_episode()
    kept += 1
    print(f"episode {kept}/{args.episodes} (seed {seed - 1}, {len(frames)} frames) {time.time() - t0:.0f}s")
ds.finalize()
print(f"done: {kept} episodes, {ds.meta.total_frames} frames -> {args.root}")
