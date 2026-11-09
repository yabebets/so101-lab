"""記事の図の共通設定（明るい背景・少ない色・Hiragino）。"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]

INK = "#1f2933"      # 文字・線
INK2 = "#52606d"     # 補助の文字
GRID = "#d9dee3"     # 薄い線
DEMO = "#9aa5b1"     # お手本（灰）
BLUE = "#2563eb"     # 主役
ORANGE = "#ea580c"   # 注意・失敗
GREEN = "#15803d"    # 正解・成功
FILL = "#f1f5f9"     # 箱の塗り
BLUE_FILL = "#dbeafe"
ORANGE_FILL = "#ffedd5"

plt.rcParams.update({
    "font.family": "Hiragino Sans",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def out(slug: str, name: str) -> Path:
    p = ROOT / "images" / slug / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save(fig, path: Path):
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT), f"{path.stat().st_size // 1024} KB")


def box(ax, x, y, w, h, text, fc=FILL, ec=INK2, fs=10, color=INK, ls="-", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            weight=weight, linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.4, ls="-", style="-|>", ms=12):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def blank(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")
