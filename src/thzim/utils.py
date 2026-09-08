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
cost nothing to import.
"""

import os
from pathlib import Path

import numpy as np
from scipy.constants import c as c_light, e as q_e, m_e

__all__ = [
    "apply_style", "output_dir", "save",
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

def apply_style(dpi=150, savefig_dpi=600):
    """Apply the project figure style (scienceplots 'science'+'ieee', no LaTeX)."""
    import matplotlib.pyplot as plt
    import scienceplots      # noqa: F401  (registers the styles)

    plt.style.use(["science", "ieee", "no-latex"])
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
