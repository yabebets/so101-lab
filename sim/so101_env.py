from pathlib import Path

import mujoco
import numpy as np

XML = Path(__file__).parent / "so101" / "pick_place.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOME = np.array([0.0, -1.6, 1.5, 0.9, 0.0, 0.0])
TRAY_XY = np.array([0.20, -0.10])
TRAY_HALF = np.array([0.055, 0.075])


class SO101PickPlaceEnv:
    def __init__(self, fps=30, image_hw=(480, 640)):
        self.m = mujoco.MjModel.from_xml_path(str(XML))
        self.d = mujoco.MjData(self.m)
        self.substeps = round(1 / fps / self.m.opt.timestep)
        self.renderer = mujoco.Renderer(self.m, *image_hw)
        self.block_adr = self.m.joint("block_free").qposadr[0]
        self.block_geom = self.m.geom("block_geom").id
        self.rng = np.random.default_rng()
        # gripper torque: 0.6..2.0 N·m all hold the 2 cm block; keep it modest so a squeeze cannot launch it
        self.m.actuator_forcerange[5] = [-1.0, 1.0]

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:6] = HOME
        self.d.ctrl[:] = HOME
        a = self.block_adr
        self.d.qpos[a : a + 3] = [self.rng.uniform(0.18, 0.26), self.rng.uniform(0.04, 0.12), self.m.geom_size[self.block_geom][2]]
        self.d.qpos[a + 3 : a + 7] = [1, 0, 0, 0]
        for _ in range(100):
            mujoco.mj_step(self.m, self.d)
        return self.observe()

    def step(self, q_target):
        lo, hi = self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1]
        self.d.ctrl[:] = np.clip(q_target, lo, hi)
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, self.d)
        return self.observe(), self.success()

    def observe(self):
        self.renderer.update_scene(self.d, "front")
        return {"image": self.renderer.render().copy(), "state": self.d.qpos[:6].copy()}

    @property
    def block_pos(self):
        return self.d.qpos[self.block_adr : self.block_adr + 3].copy()

    def success(self):
        p = self.block_pos
        return bool(np.all(np.abs(p[:2] - TRAY_XY) < TRAY_HALF) and p[2] < 0.004 + 2 * self.m.geom_size[self.block_geom][2])
