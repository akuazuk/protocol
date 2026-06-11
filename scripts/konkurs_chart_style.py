"""Пастельная палитра и оформление графиков бизнес-плана."""
from __future__ import annotations

from pathlib import Path

# Пастельные тона (неяркие, спокойные)
COLORS = [
    "#8fbc9f",  # шалфей
    "#9eb8d4",  # пыльная голубая
    "#c4b5d8",  # лаванда
    "#e8d0c4",  # персик
    "#b5d4c8",  # мята
    "#d4b5c4",  # пыльная роза
]

COLOR_PRIMARY = COLORS[0]
COLOR_SECONDARY = COLORS[1]
COLOR_ACCENT = COLORS[3]
COLOR_NEGATIVE = "#d4a8a8"
COLOR_HIGHLIGHT = COLORS[4]

BG_FIG = "#ffffff"
BG_AX = "#f9f7f4"
GRID = "#e4ebe6"
TEXT = "#3d4f48"
MUTED = "#6b7c75"


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
