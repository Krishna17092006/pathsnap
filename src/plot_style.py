"""Shared matplotlib style so every report figure looks consistent."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots in fixed order (validated colour-vision-deficiency safe for up to 3 series).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL = "#d03b3b"

PARTITION_COLOR = {"train": BLUE, "student_test": ORANGE}
PARTITION_LABEL = {"train": "Train", "student_test": "Student test"}

# Map / trajectory layers
GPS_COLOR = BLUE
ROUTE_COLOR = AQUA
PRED_COLOR = ORANGE          # classifier-only prediction
HMM_COLOR = "#6250d6"        # HMM/Viterbi prediction (violet; told apart by line style and legend, not hue alone)
NETWORK_COLOR = "#a8a69f"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
