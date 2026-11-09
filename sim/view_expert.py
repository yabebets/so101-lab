"""Open the MuJoCo viewer and replay scripted pick-and-place episodes in real time.

Run with:  .venv/bin/mjpython sim/view_expert.py [--seed N] [--loop]
Mouse: drag = rotate, right-drag = pan, scroll = zoom. Space pauses physics.
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent))
from so101_env import SO101PickPlaceEnv  # noqa: E402
from so101_expert import plan  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--loop", action="store_true", help="keep going with seed+1, +2, ...")
ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
args = ap.parse_args()

env = SO101PickPlaceEnv(image_hw=(120, 160))
env.observe = lambda: {}  # no offscreen rendering while viewing
seed = args.seed
with mujoco.viewer.launch_passive(env.m, env.d) as v:
    v.cam.lookat[:] = [0.2, 0.0, 0.05]
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.7, 160, -25
    while v.is_running():
        env.reset(seed)
        v.sync()
        try:
            traj = plan(env, env.block_pos)
        except RuntimeError as e:
            print(f"seed {seed}: {e}")
            traj = []
        succ = False
        for q in traj:
            if not v.is_running():
                break
            t0 = time.time()
            _, succ = env.step(q)
            v.sync()
            time.sleep(max(0.0, 1 / 30 / args.speed - (time.time() - t0)))
        print(f"seed {seed}: success={succ} block={env.block_pos.round(3)}")
        time.sleep(1.0)
        if not args.loop:
            seed = args.seed
        else:
            seed += 1
