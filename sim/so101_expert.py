"""Scripted pick-and-place expert for SO101PickPlaceEnv.

Geometry notes (measured in the `gripperframe` site frame, x = approach/down, z = jaw axis):
- fixed-jaw inner face is at z ~ 0, its tip 6.3 mm below the site origin
- the moving jaw is a rotating finger; its inner face is parallel to the fixed jaw only
  near q ~ 0..0.1 rad (gap 1.6..2.3 cm). Wider objects get pinched at the top edge and tip over,
  so the block must be ~2 cm wide along the jaw axis.
"""
import mujoco
import numpy as np

from so101_env import HOME, TRAY_XY

GRIPPER_OPEN, GRIPPER_CLOSED = 1.2, -0.1
DOWN = np.array([0.0, 0.0, -1.0])
TIP_BELOW_SITE = 0.0063  # fixed-jaw tip depth below the gripperframe site
TRAY_FLOOR_Z = 0.004  # top of tray floor
JAW_CLEARANCE = 0.005  # gap between fixed-jaw face and the block on approach (moving jaw pushes it onto the fixed jaw)
TIP_ABOVE_TABLE = 0.005  # jaw tip height above the table at grasp
PLACE_MARGIN = 0.002


def solve_ik(m, d, target, q_init, iters=200):
    dd = mujoco.MjData(m)
    dd.qpos[:] = d.qpos
    dd.qpos[:5] = q_init[:5]
    sid = m.site("gripperframe").id
    lo, hi = m.jnt_range[:5, 0], m.jnt_range[:5, 1]
    for _ in range(iters):
        mujoco.mj_forward(m, dd)
        approach = dd.site_xmat[sid].reshape(3, 3)[:, 0]
        err = np.concatenate([target - dd.site_xpos[sid], 0.5 * np.cross(approach, DOWN)])
        if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
            break
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, dd, jp, jr, sid)
        J = np.vstack([jp[:, :5], jr[:, :5]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        dd.qpos[:5] = np.clip(dd.qpos[:5] + 0.5 * dq, lo, hi)
    mujoco.mj_forward(m, dd)
    return dd.qpos[:5].copy(), np.linalg.norm(target - dd.site_xpos[sid])


PROFILE = "linear"  # "linear": constant speed, no zero-velocity dwell at waypoints (policies stall there); "smooth": ease in/out


def ease(t):
    return t if PROFILE == "linear" else t * t * (3 - 2 * t)


def jaw_axis_xy(m, d, q_arm, sid):
    dd = mujoco.MjData(m)
    dd.qpos[:] = d.qpos
    dd.qpos[:5] = q_arm
    mujoco.mj_forward(m, dd)
    ax = dd.site_xmat[sid].reshape(3, 3)[:, 2].copy()
    ax[2] = 0
    return ax / np.linalg.norm(ax)


def plan(env, block_xyz, hover_z=0.07, carry_z=0.07):
    """Returns a list of joint targets (6,) at control rate for one pick-and-place.

    Cartesian waypoints are interpolated in task space (IK every step) so that vertical
    moves are straight lines; joint-space interpolation bows out by several mm and bumps the block.
    """
    m, d = env.m, env.d
    sid = m.site("gripperframe").id
    half = m.geom_size[env.block_geom]  # (x, y, z) half sizes; x is along the jaw axis
    grasp_z = TIP_ABOVE_TABLE + TIP_BELOW_SITE
    place_z = TRAY_FLOOR_Z + TIP_ABOVE_TABLE + TIP_BELOW_SITE + PLACE_MARGIN
    bxy = np.asarray(block_xyz[:2])
    q_arm, _ = solve_ik(m, d, np.array([*bxy, hover_z]), HOME[:5].copy())
    # put the fixed jaw face JAW_CLEARANCE away from the block's near face
    off = jaw_axis_xy(m, d, q_arm, sid) * -(half[0] + JAW_CLEARANCE)
    gxy, txy = bxy + off[:2], TRAY_XY + off[:2]
    seq = [  # (name, xyz target or None, gripper target, steps, interpolate in task space?)
        ("above", np.array([*gxy, hover_z]), GRIPPER_OPEN, 40, False),
        ("grasp", np.array([*gxy, grasp_z]), GRIPPER_OPEN, 40, True),
        ("settle", None, GRIPPER_OPEN, 6, False),
        ("close", None, GRIPPER_CLOSED, 40, False),
        ("lift", np.array([*gxy, carry_z]), GRIPPER_CLOSED, 40, True),
        ("over_tray", np.array([*txy, carry_z]), GRIPPER_CLOSED, 90, False),
        ("lower", np.array([*txy, place_z]), GRIPPER_CLOSED, 40, True),
        ("release", None, GRIPPER_OPEN, 25, False),
        ("up", np.array([*txy, carry_z]), GRIPPER_OPEN, 30, True),
    ]
    traj, cur = [], HOME.copy()
    q_arm, cur_xyz = HOME[:5].copy(), None
    for name, xyz, grip, n, cartesian in seq:
        if xyz is not None and cartesian and cur_xyz is not None:
            for i in range(1, n + 1):
                p = cur_xyz + (xyz - cur_xyz) * ease(i / n)
                q_arm, e = solve_ik(m, d, p, q_arm, iters=50)
                if e > 5e-3:
                    raise RuntimeError(f"IK failed at {name} {p}, residual {e:.3f}")
                g = cur[5] + (grip - cur[5]) * ease(i / n)
                traj.append(np.concatenate([q_arm, [g]]))
            cur, cur_xyz = traj[-1].copy(), xyz
            continue
        if xyz is not None:
            q_arm, e = solve_ik(m, d, xyz, q_arm)
            if e > 5e-3:
                raise RuntimeError(f"IK failed at {name} {xyz}, residual {e:.3f}")
            cur_xyz = xyz
        goal = np.concatenate([q_arm, [grip]])
        for i in range(1, n + 1):
            traj.append(cur + (goal - cur) * ease(i / n))
        cur = goal
    return traj
