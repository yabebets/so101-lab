"""記事「SO-ARM101 の MuJoCo モデルでシミュレーション上の模倣学習」の図。
.venv/bin/python tools/figures/mujoco_sim_act.py
"""
import sys

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from common import (ROOT, INK, INK2, GRID, DEMO, BLUE, ORANGE, GREEN, FILL, BLUE_FILL, ORANGE_FILL,
                    out, save, arrow)

sys.path.insert(0, str(ROOT / "sim"))
import so101_expert  # noqa: E402
from so101_env import SO101PickPlaceEnv  # noqa: E402

SLUG = "so101-mujoco-sim-act"
SEG_JA = {"above": "上空へ", "grasp": "下ろす", "settle": "", "close": "閉じる", "lift": "持ち上げ",
          "over_tray": "トレイへ", "lower": "下ろす", "release": "開く", "up": "上げる"}
SEG_N = [("above", 40), ("grasp", 40), ("settle", 6), ("close", 40), ("lift", 40),
         ("over_tray", 90), ("lower", 40), ("release", 25), ("up", 30)]


def tip_speed(env, traj, fps=30):
    m = env.m; d = mujoco.MjData(m); d.qpos[:] = env.d.qpos
    sid = m.site("gripperframe").id
    pts = []
    for q in traj:
        d.qpos[:5] = q[:5]; mujoco.mj_kinematics(m, d); pts.append(d.site_xpos[sid].copy())
    pts = np.array(pts)
    return np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=1)] * fps * 100  # cm/s


def ease_vs_constant():
    env = SO101PickPlaceEnv(image_hw=(48, 64))
    speeds = {}
    for prof in ("smooth", "linear"):
        so101_expert.PROFILE = prof
        env.reset(seed=3)
        speeds[prof] = tip_speed(env, so101_expert.plan(env, env.block_pos))
    fig, axs = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
    t = np.arange(len(speeds["smooth"])) / 30
    bounds = np.cumsum([0] + [n for _, n in SEG_N]) / 30
    for ax, prof, title, color in [(axs[0], "smooth", "学習 1 回目：ease-in/out（なめらかに加速・減速）", ORANGE),
                                   (axs[1], "linear", "学習 2 回目：等速", BLUE)]:
        for k, (name, _) in enumerate(SEG_N):
            if k % 2 == 0:
                ax.axvspan(bounds[k], bounds[k + 1], color=FILL, lw=0)
        ax.plot(t, speeds[prof], color=color, lw=1.8)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_ylabel("手先の速さ\n（cm/秒）", fontsize=9)
        ax.set_ylim(0, max(speeds["smooth"].max(), speeds["linear"].max()) * 1.18)
    for k, (name, _) in enumerate(SEG_N):
        if SEG_JA[name]:
            axs[0].text((bounds[k] + bounds[k + 1]) / 2, axs[0].get_ylim()[1] * 0.97, SEG_JA[name],
                        ha="center", va="top", fontsize=8, color=INK2)
    x0 = bounds[1]
    axs[0].annotate("真上で速さがゼロ\n→ 方策がここで\n　 止まってしまう",
                    xy=(x0 + 0.05, 0.4), xytext=(2.85, 0.9), fontsize=8.5, color=ORANGE,
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1))
    axs[0].plot(x0, 0.3, "o", color=ORANGE, ms=7)
    axs[1].annotate("区間の切れ目でも止まらない", xy=(x0, speeds["linear"][int(x0 * 30)]),
                    xytext=(1.6, 6.6), fontsize=8.5, color=BLUE,
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1))
    axs[1].set_xlabel("時間（秒）  ※ seed 3 のデモ。止まって見える区間はグリッパーの開閉だけをしているところ", fontsize=8.5)
    save(fig, out(SLUG, "fig_ease_vs_constant.png"))


def dart():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    n = 14
    x = np.linspace(0, 10, n)
    plan_y = np.linspace(3, 0.4, n)                       # 教師が計画した道すじ（ノイズなし）
    exe = plan_y + np.r_[0, rng.normal(0, 0.22, n - 1)]   # 実行した道すじ（ノイズ入り）
    ax.plot(x, plan_y, color=GREEN, lw=1.8, ls="--", label="教師の計画（ノイズなし）")
    ax.plot(x, exe, color=BLUE, lw=2, marker="o", ms=5, label="実際に動かした道すじ（関節の目標にノイズを足す）")
    for i in range(1, n - 1, 2):
        arrow(ax, x[i], exe[i], x[i + 1], plan_y[i + 1], color=ORANGE, lw=1.3)
    ax.plot([], [], color=ORANGE, lw=1.3, label="記録する行動（その状態から、ノイズなしの次の目標へ）")
    j = int(np.argmax(np.abs(exe - plan_y)[:-1]))
    ax.annotate("計画から少しずれた状態", xy=(x[j], exe[j]), xytext=(x[j] - 1.2, exe[j] + 1.0), fontsize=8.5, color=BLUE,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0))
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.text(0.0, -0.55, "方策は「計画からずれたところから、どう戻るか」も学べる。\n"
            "ノイズなしのデモだけだと、ずれた状態はデータに一度も出てこない（誤差の累積に弱い）",
            fontsize=9, color=INK2, va="top")
    ax.set_xlim(-0.3, 10.5); ax.set_ylim(-1.6, 4.4); ax.set_axis_off()
    ax.set_title("DART 方式のノイズ注入（模式図）", fontsize=11, color=INK, loc="left")
    save(fig, out(SLUG, "fig_dart.png"))


def ik_iterations():
    L = np.array([11.6, 13.5])  # cm（SO-101 の上腕・前腕くらい）

    def fk(q):
        p1 = L[0] * np.array([np.cos(q[0]), np.sin(q[0])])
        return p1, p1 + L[1] * np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1])])

    def jac(q):
        s1, c1, s12, c12 = np.sin(q[0]), np.cos(q[0]), np.sin(q.sum()), np.cos(q.sum())
        return np.array([[-L[0] * s1 - L[1] * s12, -L[1] * s12], [L[0] * c1 + L[1] * c12, L[1] * c12]])

    target = np.array([15.0, 12.0])
    q = np.array([1.3, -1.9]); qs, errs = [q.copy()], []
    for _ in range(12):
        e = target - fk(q)[1]; errs.append(np.linalg.norm(e))
        q = q + 0.6 * np.linalg.pinv(jac(q)) @ e; qs.append(q.copy())
    errs.append(np.linalg.norm(target - fk(q)[1]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    K = next(i for i, e in enumerate(errs) if e < 0.5)
    for k, qk in enumerate(qs[:K + 1]):
        p1, p2 = fk(qk); al = 0.3 + 0.7 * k / K
        col = BLUE if k == K else DEMO
        a.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], "-o", color=col, alpha=al, lw=2.2 if k == K else 1.2, ms=4)
        if 0 < k < K:
            a.text(p2[0] + 0.5, p2[1] - 0.4, str(k), fontsize=7.5, color=INK2)
    a.plot(*target, "*", color=GREEN, ms=16); a.text(target[0] + 0.8, target[1] + 0.5, "目標", color=GREEN, fontsize=9)
    a.plot(0, 0, "s", color=INK, ms=8)
    p2 = fk(qs[0])[1]; a.text(p2[0] + 0.6, p2[1] - 0.6, "最初の姿勢", fontsize=8.5, color=INK2)
    a.set_aspect("equal"); a.set_xlim(-3, 26); a.set_ylim(-3, 22); a.set_axis_off()
    a.set_title("手先のずれを関節の動きに直して、少しずつ近づける", fontsize=10, color=INK, loc="left")
    fig.text(0.02, -0.04, "Δq = $J^{+}$ (目標 − 今の手先)　　$J^{+}$：ヤコビアンの疑似逆行列\n"
           "2 関節の平面アームでの例。SO-101 では 5 関節で、手先の向きの条件も足して同じことをする",
           fontsize=8.5, color=INK2, va="top")
    b.semilogy(range(len(errs)), np.maximum(errs, 1e-5), "-o", color=BLUE, ms=4)
    b.axhline(0.5, color=ORANGE, lw=1, ls="--"); b.text(len(errs) - 1, 0.6, "教師が許す誤差 5 mm", ha="right",
                                                          va="bottom", fontsize=8, color=ORANGE)
    b.set_xlabel("反復の回数"); b.set_ylabel("手先と目標の距離（cm、対数）")
    b.set_title("反復ごとのずれ", fontsize=10, color=INK, loc="left")
    save(fig, out(SLUG, "fig_ik_iterations.png"))


if __name__ == "__main__":
    ik_iterations()
    ease_vs_constant()
    dart()
