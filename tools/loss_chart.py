import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Arial Unicode MS"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
rows = list(csv.DictReader(open(sys.argv[1])))
x = [int(r["step"]) for r in rows]; y = [float(r["loss"]) for r in rows]
SURF, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.plot(x, y, color=BLUE, linewidth=2, solid_capstyle="round")
ax.set_yscale("log")
ax.set_xlim(0, 30000)
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xticks(range(0, 30001, 5000)); ax.set_xticklabels([f"{v//1000}k" if v else "0" for v in range(0, 30001, 5000)])
ax.grid(axis="y", which="major", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9, length=0)
ax.set_xlabel("学習ステップ", color=INK2, fontsize=9)
ax.set_ylabel("loss（対数軸）", color=INK2, fontsize=9)
ax.set_title("ACT 学習の loss 推移（Colab A100, 30k steps, 32分37秒）", color=INK, fontsize=11, loc="left", pad=12)
ax.annotate(f"{y[0]:.2f}", (x[0], y[0]), xytext=(6, 0), textcoords="offset points", color=INK2, fontsize=9, va="center")
ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(-4, 10), textcoords="offset points", color=INK2, fontsize=9, ha="right")
fig.tight_layout(); fig.savefig(sys.argv[2], facecolor=SURF)
