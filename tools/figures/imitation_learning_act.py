"""記事「実機で模倣学習：テレオペから ACT 学習まで」の図。
.venv/bin/python tools/figures/imitation_learning_act.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon

from common import (INK, INK2, GRID, DEMO, BLUE, ORANGE, GREEN, FILL, BLUE_FILL, ORANGE_FILL,
                    out, save, box, arrow, blank)

SLUG = "so101-imitation-learning-act"
rng = np.random.default_rng(3)


def bc_weaknesses():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    # 左：誤差の累積
    t = np.linspace(0, 1, 200)
    base = 0.35 * np.sin(np.pi * t)
    a.fill_between(t, base - 0.07, base + 0.07, color=GRID, alpha=0.7, lw=0)
    for _ in range(7):
        a.plot(t, base + rng.normal(0, 0.02) + 0.025 * np.sin(6 * t + rng.uniform(0, 6)), color=DEMO, lw=1)
    drift = 0.62 * t ** 2.2
    a.plot(t, base - drift, color=BLUE, lw=2.4)
    a.plot(0, 0, "o", color=INK, ms=6); a.text(0.0, -0.06, "スタート", ha="left", va="top", fontsize=9, color=INK2)
    a.plot(1, 0, "*", color=GREEN, ms=14); a.text(1.0, -0.07, "ゴール", ha="center", va="top", fontsize=9, color=GREEN)
    a.text(0.5, 0.47, "お手本が通った範囲", ha="center", fontsize=9, color=INK2)
    a.annotate("小さなずれが積み重なり、\nお手本で見たことのない\n状態に迷い込む", xy=(0.8, (base - drift)[159]),
               xytext=(0.12, -0.5), fontsize=9, color=BLUE,
               arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    a.set_title("① 誤差の累積（covariate shift）", fontsize=11, color=INK, loc="left")
    a.set_xlim(-0.05, 1.08); a.set_ylim(-0.7, 0.6); a.set_axis_off()
    # 右：お手本のばらつき
    b.add_patch(Circle((0.5, 0), 0.14, fc=FILL, ec=INK2, lw=1.2))
    b.text(0.5, 0, "障害物", ha="center", va="center", fontsize=9, color=INK2)
    for s in (1, -1):
        for _ in range(4):
            amp = 0.3 + rng.normal(0, 0.03)
            b.plot(t, s * amp * np.sin(np.pi * t), color=DEMO, lw=1)
    b.text(0.5, 0.40, "右から回るお手本", ha="center", fontsize=9, color=INK2)
    b.text(0.5, -0.43, "左から回るお手本", ha="center", fontsize=9, color=INK2)
    b.plot(t[t < 0.36], 0 * t[t < 0.36], color=ORANGE, lw=2.4, ls="--")
    b.text(0.355, 0.0, "×", color=ORANGE, fontsize=22, ha="center", va="center", weight="bold")
    b.annotate("平均すると\nまっすぐ突っ込む", xy=(0.22, 0.0), xytext=(0.02, 0.2), fontsize=9, color=ORANGE,
               arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
    b.plot(0, 0, "o", color=INK, ms=6); b.plot(1, 0, "*", color=GREEN, ms=14)
    b.set_title("② お手本のばらつき（マルチモーダル性）", fontsize=11, color=INK, loc="left")
    b.set_xlim(-0.05, 1.08); b.set_ylim(-0.6, 0.6); b.set_axis_off()
    save(fig, out(SLUG, "fig_bc_weaknesses.png"))


def calibration_mapping():
    fig, ax = plt.subplots(figsize=(10, 4.4)); blank(ax)
    ax.set_xlim(-2.6, 10.4); ax.set_ylim(-1.3, 5.2)

    def bar(y, x0, x1, lo, hi, label, ticks, color=BLUE_FILL):
        ax.add_patch(Rectangle((0, y), 10, 0.5, fc="white", ec=GRID, lw=1))
        ax.add_patch(Rectangle((x0, y), x1 - x0, 0.5, fc=color, ec=INK2, lw=1))
        ax.text(-0.2, y + 0.25, label, ha="right", va="center", fontsize=10, color=INK)
        ax.text(x0, y - 0.12, lo, ha="center", va="top", fontsize=8.5, color=INK2)
        ax.text(x1, y - 0.12, hi, ha="center", va="top", fontsize=8.5, color=INK2)
        for x, s in ticks:
            ax.text(x, y + 0.62, s, ha="center", fontsize=8.5, color=INK2)

    # リーダー（生の値は例）
    bar(4.2, 1.5, 7.5, "range_min", "range_max", "リーダー\n（生の値）", [])
    # 正規化
    bar(2.4, 0, 10, "0 %", "100 %", "正規化した値\n（割合）", [], color="#eef2ff")
    # フォロワー
    bar(0.6, 2.6, 6.9, "1083", "2837", "フォロワー\n（生の値）", [])
    p = 0.17
    xl, xn, xf = 1.5 + p * 6, p * 10, 2.6 + p * 4.3
    for (x0, y0, x1, y1) in [(xl, 4.2, xn, 2.9), (xn, 2.4, xf, 1.1)]:
        arrow(ax, x0, y0, x1, y1, color=BLUE, lw=1.6)
    for x, y in [(xl, 4.45), (xn, 2.65), (xf, 0.85)]:
        ax.plot(x, y, "o", color=BLUE, ms=7)
    ax.text(xn + 0.25, 2.65, "17 %", va="center", fontsize=10, color=BLUE, weight="bold")
    ax.text(5.1, 4.92, "リーダーで 17 % まで閉じる", ha="center", fontsize=9.5, color=INK)
    ax.text(7.2, 1.45, "フォロワーにも、自分の範囲の\n17 % の位置が指令される", ha="left", fontsize=9.5, color=INK)
    # 物理的に届かない範囲
    ax.add_patch(Rectangle((2.6, 0.6), 0.55, 0.5, fc=ORANGE_FILL, ec=ORANGE, lw=1.2, hatch="///"))
    ax.annotate("キャリブレーションが実機とずれていると、\nこの辺りは物理的に届かない位置になる\n→ サーボが押し続けて過負荷に",
                xy=(2.85, 0.6), xytext=(4.0, -0.05), fontsize=8.5, color=ORANGE, va="top",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1))
    save(fig, out(SLUG, "fig_calibration_mapping.png"))


def camera_shift():
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))

    def frame(ax, dx, title, note):
        ax.add_patch(Rectangle((0, 0), 4, 3, fc="#fbf7f0", ec=INK2, lw=1.2))
        ax.add_patch(Rectangle((1.9 + dx, 0.8), 1.3, 1.0, fc="#c08457", ec=INK2, lw=1, alpha=0.8))  # トレイ
        ax.text(2.55 + dx, 1.3, "トレイ", ha="center", va="center", fontsize=8, color="white")
        ax.add_patch(Rectangle((0.9 + dx, 1.9), 0.35, 0.22, fc=BLUE, ec=INK, lw=1))  # ブロック
        ax.plot([1.075 + dx] * 2, [0, 3], color=BLUE, lw=0.8, ls=":")
        ax.text(2, 3.25, title, ha="center", fontsize=10, color=INK)
        ax.text(2, -0.35, note, ha="center", va="top", fontsize=8.5, color=INK2)
        ax.set_xlim(-0.2, 4.2); ax.set_ylim(-1.0, 3.6); blank(ax)

    frame(axs[0], 0, "収録したときの画像", "ブロックは画面の左のこの位置")
    frame(axs[1], -0.45, "カメラが少し動いたあとの画像", "同じ場所のブロックが、画面では左に写る")
    axs[1].plot([1.075] * 2, [0, 3], color=DEMO, lw=0.8, ls=":")
    arrow(axs[1], 1.0, 2.35, 0.68, 2.35, color=ORANGE, lw=1.4)
    # 机を上から見た図
    ax = axs[2]; blank(ax)
    ax.add_patch(Rectangle((0, 0), 4.4, 3, fc="#fbf7f0", ec=INK2, lw=1.2))
    ax.add_patch(Rectangle((0.1, 0.9), 0.7, 0.9, fc=FILL, ec=INK2))
    ax.text(0.45, 1.35, "アーム", ha="center", va="center", fontsize=8, color=INK2, rotation=90)
    ax.plot(2.4, 2.0, "s", color=BLUE, ms=11); ax.text(2.6, 2.0, "本当の位置", ha="left", va="center", fontsize=8.5, color=BLUE)
    ax.plot(1.75, 2.0, "x", color=ORANGE, ms=12, mew=2.5)
    ax.text(1.75, 2.3, "手を伸ばす先", ha="center", fontsize=8.5, color=ORANGE)
    arrow(ax, 0.8, 1.5, 1.65, 1.95, color=ORANGE, lw=1.4)
    ax.text(2.2, 3.25, "机を上から見ると", ha="center", fontsize=10, color=INK)
    ax.text(2.2, -0.35, "モデルは「画面のどこに写っているか」で\n位置を覚えているので、狙いがずれる", ha="center", va="top",
            fontsize=8.5, color=INK2)
    ax.set_xlim(-0.2, 4.6); ax.set_ylim(-1.0, 3.6)
    save(fig, out(SLUG, "fig_camera_shift.png"))


def action_chunking():
    fig, ax = plt.subplots(figsize=(10, 3.9))
    T = 200
    rows = [("1 ステップずつ予測", 1, 1), ("ACT（n_action_steps = 100）", 100, 100), ("ACT（n_action_steps = 25）", 25, 100)]
    for i, (label, step, chunk) in enumerate(rows):
        y = 2 - i
        ax.text(-6, y, label, ha="right", va="center", fontsize=10, color=INK)
        if step == 1:
            ax.add_patch(Rectangle((0, y - 0.13), T, 0.26, fc=BLUE_FILL, ec="none"))
            ax.plot(np.arange(0, T, 1), np.full(T, y + 0.25), "|", color=BLUE, ms=6, mew=0.6)
            ax.text(T + 4, y, "毎ステップ（1/30 秒ごと）に画像を見て、\n次の 1 ステップだけを決める", va="center", fontsize=8.5, color=INK2)
            continue
        if step < chunk:
            ax.add_patch(Rectangle((step, y - 0.42), chunk - step, 0.16, fc=GRID, ec="none"))
            ax.text(chunk + 2, y - 0.34, "最初の予測のうち、使わずに捨てる 75 ステップ", va="center", fontsize=8, color=INK2)
        for k, t0 in enumerate(range(0, T, step)):
            ax.add_patch(Rectangle((t0, y - 0.13), min(step, T - t0), 0.26, fc=BLUE_FILL, ec=BLUE, lw=0.8))
            ax.plot(t0, y + 0.27, "v", color=BLUE, ms=7)
        note = ("100 ステップ（約 3.3 秒）分をまとめて予測し、\n全部実行してから次を予測" if step == 100
                else "100 ステップ分を予測するが、先頭の 25 ステップ\nだけ実行して予測し直す（約 0.8 秒ごと）")
        ax.text(T + 4, y, note, va="center", fontsize=8.5, color=INK2)
    ax.plot([], [], "v", color=BLUE, label="画像を見て予測する");
    ax.add_patch(Rectangle((0, -10), 1, 1, fc=BLUE_FILL, ec=BLUE, label="実行する分"))
    ax.add_patch(Rectangle((0, -10), 1, 1, fc=GRID, ec="none", label="予測したが捨てる分"))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False, fontsize=9)
    ax.set_xlim(-2, T + 2); ax.set_ylim(-0.7, 2.6)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
    ax.set_xlabel("制御ステップ（30 fps）")
    save(fig, out(SLUG, "fig_action_chunking.png"))


def act_architecture():
    fig, ax = plt.subplots(figsize=(12, 5.4)); blank(ax)
    ax.set_xlim(0, 25); ax.set_ylim(0, 10.6)
    # 学習時だけの CVAE エンコーダ
    ax.add_patch(Rectangle((0.2, 6.2), 9.3, 4.1, fc="none", ec=ORANGE, ls="--", lw=1.2))
    ax.text(0.4, 9.95, "学習のときだけ使う部分", fontsize=9, color=ORANGE, va="center")
    box(ax, 0.6, 7.8, 4.0, 1.5, "お手本の行動列\n100 ステップ × 6 関節", fc=ORANGE_FILL, ec=ORANGE, fs=9)
    box(ax, 0.6, 6.5, 4.0, 1.0, "関節角（6）", fc=FILL, fs=9)
    box(ax, 5.5, 7.0, 3.7, 1.8, "CVAE エンコーダ\n（Transformer）", fc=ORANGE_FILL, ec=ORANGE, fs=9.5)
    arrow(ax, 4.6, 8.55, 5.5, 8.1); arrow(ax, 4.6, 7.0, 5.5, 7.6)
    box(ax, 11.0, 7.3, 3.0, 1.2, "スタイル z\n（32 次元）", fc=ORANGE_FILL, ec=ORANGE, fs=9.5)
    arrow(ax, 9.2, 7.9, 11.0, 7.9)
    ax.text(12.5, 9.0, "推論のときは z = 0\n（平均的なやり方）", ha="center", fontsize=8.5, color=ORANGE)
    ax.text(16.6, 7.9, "z がやり方の違いを吸収するので、\n複数のお手本を平均しなくてすむ", ha="left", va="center", fontsize=8.5, color=ORANGE)
    # 本体
    box(ax, 0.6, 3.6, 4.0, 1.4, "カメラ画像\n640 × 480", fc=FILL, fs=9.5)
    box(ax, 5.5, 3.6, 3.7, 1.4, "ResNet18\n（ImageNet で事前学習）", fc=BLUE_FILL, ec=BLUE, fs=8.5)
    arrow(ax, 4.6, 4.3, 5.5, 4.3)
    box(ax, 0.6, 1.4, 4.0, 1.1, "関節角（6）", fc=FILL, fs=9.5)
    box(ax, 10.6, 1.2, 3.8, 5.2, "Transformer\nエンコーダ（4 層）\n\n↓\n\nTransformer\nデコーダ（1 層）\n\ndim_model = 512", fc=BLUE_FILL, ec=BLUE, fs=9.5)
    arrow(ax, 9.2, 4.3, 10.6, 4.3); ax.text(9.9, 4.55, "画像の特徴", ha="center", fontsize=8, color=INK2)
    arrow(ax, 4.6, 1.95, 10.6, 1.95)
    arrow(ax, 12.5, 7.3, 12.5, 6.4, color=ORANGE)
    box(ax, 16.0, 2.6, 4.6, 2.4, "行動列\n100 ステップ × 6 関節\n（約 3.3 秒分）", fc="#dcfce7", ec=GREEN, fs=9.5)
    arrow(ax, 14.4, 3.8, 16.0, 3.8)
    arrow(ax, 20.6, 3.8, 21.8, 3.8)
    ax.text(22.0, 3.8, "ロボットへ", va="center", fontsize=9.5, color=INK)
    ax.text(18.3, 2.2, "学習では、お手本の行動列との\nずれ（L1）を小さくする", ha="center", va="top", fontsize=8.5, color=INK2)
    save(fig, out(SLUG, "fig_act_architecture.png"))


if __name__ == "__main__":
    bc_weaknesses()
    calibration_mapping()
    camera_shift()
    action_chunking()
    act_architecture()
