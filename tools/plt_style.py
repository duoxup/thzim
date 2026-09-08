#!/usr/bin/env python3
"""Figure style and output helpers for the standalone tools.

The tools-side copy of `thzim.utils` (`STYLE`, `apply_style`, `output_dir`,
`save`), kept here so the sizing maps run without installing the package.
Keep the two in step. Needs only matplotlib.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt

# The project figure style, written out by hand so that no style package is
# needed: a compact serif look for two-column papers (inward ticks on all four
# sides, minor ticks, thin axes, a muted colour cycle, no grid, no legend
# frame). Colour carries the plane in every figure and the linestyle the curve
# kind, so the cycle deliberately varies COLOUR ONLY -- scripts still pin
# `ls=` explicitly, which keeps them robust to any later style change.
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif",
                   "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "axes.prop_cycle": "cycler('color', ['#0C5DA5', '#00B945', '#FF9500', "
                       "'#FF2C00', '#845B97', '#474747', '#9E9E9E'])",
    "axes.formatter.use_mathtext": True,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "patch.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "legend.frameon": False,
    "figure.figsize": (3.3, 2.5),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,
}


def apply_style(dpi=150, savefig_dpi=600):
    """Apply the project figure style (`STYLE`, plus the two dpi settings)."""
    plt.rcdefaults()
    plt.rcParams.update(STYLE)
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = savefig_dpi


def output_dir(repo_root, *parts):
    """Return ``THZIM_OUTPUT_DIR`` or the repository-local ``outputs`` path."""
    configured = os.getenv("THZIM_OUTPUT_DIR")
    root = Path(configured).expanduser() if configured else Path(repo_root) / "outputs"
    return root.joinpath(*parts)


def save(fig, path=None, fname=None):
    """Write ``fig`` as PDF and PNG, or do nothing when ``path`` is ``None``."""
    if path is None:
        return
    if fname is None:
        raise ValueError("fname is required when path is not None")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{fname}.pdf")
    fig.savefig(path / f"{fname}.png")
