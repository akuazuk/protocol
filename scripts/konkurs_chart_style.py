"""Пастельная палитра и оформление графиков бизнес-плана."""
from __future__ import annotations

from pathlib import Path

# Пастель с чуть большим контрастом
COLORS = [
    "#4d8f73",  # шалфей
    "#5a85b5",  # голубой
    "#7a68a8",  # лаванда
    "#c4957a",  # персик
    "#5a9e82",  # мята
    "#a86b88",  # роза
]

COLOR_PRIMARY = COLORS[0]
COLOR_SECONDARY = COLORS[1]
COLOR_ACCENT = COLORS[3]
COLOR_NEGATIVE = "#c47878"
COLOR_HIGHLIGHT = COLORS[4]

BG_FIG = "#ffffff"
BG_AX = "#f3f1ee"
GRID = "#c5d1cb"
TEXT = "#24332f"
MUTED = "#4d5e58"


def apply_rc(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": BG_FIG,
            "axes.facecolor": BG_AX,
            "axes.edgecolor": "#d0ddd6",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "grid.color": GRID,
            "grid.alpha": 0.85,
            "grid.linewidth": 0.6,
        }
    )


def style_ax(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0ddd6")
    ax.spines["bottom"].set_color("#d0ddd6")
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.75, linestyle="-", linewidth=0.5, color=GRID)
        ax.set_axisbelow(True)


def save_fig(fig, path: Path, dpi: int = 170) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG_FIG, edgecolor="none")
    fig.clf()
