"""Evaluate a trained LeRobot policy in SO101PickPlaceEnv.

Usage:  .venv/bin/python sim/eval_policy.py --policy outputs/train/so101_sim_act/checkpoints/last/pretrained_model \
            --episodes 20 [--video sim/eval.mp4] [--device mps]
Seeds 0..episodes-1 are used (the dataset was collected with seeds >= 1000, so these are unseen).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from so101_env import SO101PickPlaceEnv  # noqa: E402

from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True, help="pretrained_model directory")
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--max-steps", type=int, default=450)
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
ap.add_argument("--height", type=int, default=240)
ap.add_argument("--width", type=int, default=320)
ap.add_argument("--video", type=Path, default=None, help="write all episodes to this mp4")
ap.add_argument("--temporal-ensemble", type=float, default=None, help="ACT temporal ensembling coeff (e.g. 0.01); re-plans every step")
args = ap.parse_args()

cfg = PreTrainedConfig.from_pretrained(args.policy)
cfg.device = args.device
if args.temporal_ensemble is not None:
    cfg.temporal_ensemble_coeff, cfg.n_action_steps = args.temporal_ensemble, 1
policy = get_policy_class(cfg.type).from_pretrained(args.policy, config=cfg)
policy.to(args.device).eval()
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.policy, preprocessor_overrides={"device_processor": {"device": args.device}})
print(f"loaded {cfg.type} policy on {args.device}: chunk_size={getattr(cfg, 'chunk_size', '?')} n_action_steps={getattr(cfg, 'n_action_steps', '?')} temporal_ensemble={getattr(cfg, 'temporal_ensemble_coeff', None)}")

env = SO101PickPlaceEnv(image_hw=(args.height, args.width))
ff = None
if args.video:
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}", "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(args.video)], stdin=subprocess.PIPE)

results = []
for ep in range(args.episodes):
    seed = args.seed0 + ep
    obs = env.reset(seed)
    policy.reset()
    success, t0 = False, time.time()
    for t in range(args.max_steps):
        batch = {
            "observation.state": torch.from_numpy(obs["state"]).float(),
            "observation.images.front": torch.from_numpy(obs["image"]).permute(2, 0, 1).float() / 255.0,
        }
        with torch.no_grad():
            action = post(policy.select_action(pre(batch)))
        obs, success = env.step(action.squeeze(0).cpu().numpy())
        if ff:
            ff.stdin.write(obs["image"].tobytes())
        if success:
            break
    results.append(success)
    print(f"episode {ep} (seed {seed}): {'SUCCESS' if success else 'fail'} after {t + 1} steps, block={env.block_pos.round(3)}, {time.time() - t0:.1f}s", flush=True)
if ff:
    ff.stdin.close()
    ff.wait()
print(f"success rate: {sum(results)}/{len(results)} = {100 * np.mean(results):.0f}%")
