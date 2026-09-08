"""Plotting helpers and shared small utilities.

Two unrelated groups live here on purpose, because both are one-liners that
every other module would otherwise redefine:

* **figure output** -- `apply_style` / `output_dir` / `save` are the
  package-side copy of `tools/plt_style.py`; the standalone sizing tools keep
  their own so they stay runnable without installing thzim, while the repro/
  scripts and the demo tools (which need the package anyway) import from here.
* **beam rigidity** -- `M_E_GEV`, `brho`, `k_of_g`, `g_of_k`. Every design
  module needs the gradient [T/m] <-> geometric strength [1/m^2] conversion,
  and it is the ONLY place beam energy enters a linear-optics calculation, so
  it is worth having exactly one definition of it. `thzim.chicane` and
  `thzim.dogleg` re-export `M_E_GEV` and `brho` for backwards compatibility.

matplotlib is imported lazily inside `apply_style`, so the rigidity helpers
cost nothing to import. The style itself is the `STYLE` dict of rcParams,
written out by hand rather than taken from a style package.
"""

import os
from pathlib import Path

import numpy as np
from scipy.constants import c as c_light, e as q_e, m_e

__all__ = [
    "STYLE", "apply_style", "output_dir", "save",
    "M_E_GEV", "brho", "k_of_g", "g_of_k",
]

M_E_GEV = m_e * c_light**2 / q_e * 1e-9        # electron rest energy [GeV]


# ------------------------------- beam rigidity -------------------------------

def brho(energy_gev):
    """Magnetic rigidity B rho [T m] from the TOTAL energy [GeV]."""
    return np.sqrt(energy_gev**2 - M_E_GEV**2) * 1e9 / c_light


def k_of_g(g, energy_gev):
    """Geometric quad strength k1 [1/m^2] from a gradient g [T/m]."""
    return g / brho(energy_gev)


def g_of_k(k, energy_gev):
    """Gradient g [T/m] from a geometric quad strength k1 [1/m^2].

    The inverse of `k_of_g`. Which of the two is "the" knob is a design choice:
    k1 is what the optics sees and is energy-free, g is what the power supply
    delivers. `thzim.dogleg` solves in k1, `thzim.triplet` solves in g -- see
    that module's docstring for why.
    """
    return k * brho(energy_gev)


# -------------------------------- figure style --------------------------------

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
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    plt.rcParams.update(STYLE)
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = savefig_dpi


def output_dir(repo_root, *parts):
    """Return a portable output path, optionally overridden by the environment.

    ``THZIM_OUTPUT_DIR`` names the output root.  When it is unset, outputs live
    under ``<repo_root>/outputs``.  Subdirectories in ``parts`` keep figures
    from different tools and operating points separate.
    """
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
