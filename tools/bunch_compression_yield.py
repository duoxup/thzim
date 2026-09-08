#!/usr/bin/env python3
r"""Bunch-compressor yield map: what R56 buys you, at a given chirp.

The companion tool ``bunch_compression_scan.py`` answers "what correlated spread
must the injector deliver to hit a target current?". This one asks nothing about
targets: the correlated spread is FIXED, and the map simply reports what a given
R56 compresses the bunch to.

Grid axes: rms bunch length sigma_z,b BEFORE compression  x  compressor R56.
Colour map: peak current (or sigma_z,b) AFTER compression.

At fixed relative spread the algebra collapses to one line. With h = spread /
sigma_z0 and the z-convention map M = 1 + h R56,

    sigma_z,b(after) = |M| sigma_z0 = |sigma_z0 + spread * R56|

so the length after the compressor is LINEAR in both axes, and full compression
sits on the straight line sigma_z0 = -spread * R56 through the origin. Crossing
it flips under- into over-compression, which is why every panel shows a ridge:
the peak current rises to a singular line and falls away again.

Peak current follows from the profile model, I_peak = k Q c / sigma_z, with k
set by the current shape (1/sqrt(2 pi) Gaussian, 3/(4 sqrt 5) inverted
parabola); peak_current() lives with the profile models in the companion tool. On the ridge this diverges, so the colour scale is capped per panel
and drawn with extend="max": read the top band as "past the useful working
point", not as a prediction. The linear model has no T566, no LSC and no
CSR/SC, all of which matter most exactly there.

Every beam input is a free parameter in the settings block -- charge, correlated
spread, and the working point (bunch length before compression, R56). They are
seeded with the values measured from data/OP*_50k.dist, but nothing here reads a
distribution, so the tool answers "what if" questions as readily as "what is".
To re-derive the seeds after the beams change, use measure_beam() from
bunch_compression_scan.py: charge, and spread_cor = cor_pz / mean(pz).

Layout: physics -> ``__main__`` (user settings, compute, plot). Shares the
profile models and peak_current() with bunch_compression_scan.py (same directory, so a plain import
works). Figures are shown, not saved.
"""

import numpy as np
import matplotlib.pyplot as plt
import plt_style as ps
from pathlib import Path
from bunch_compression_scan import PROFILES, peak_current


# --------------------------------- physics ---------------------------------

def compressed_sigma_z(sigma_z0, r56, spread, convention="z"):
    """rms bunch length after the compressor, at fixed relative spread.

    `spread` is the signed correlated dp/p0; the chirp it implies, h = spread /
    sigma_z0, is what enters the longitudinal map.
    """
    if convention == "z":
        return np.abs(sigma_z0 + spread * r56)
    if convention == "tau":
        return np.abs(sigma_z0 - spread * r56)
    raise ValueError(f"convention must be 'z' or 'tau', got {convention!r}")


def full_compression_r56(sigma_z0, spread, convention="z"):
    """R56 at which the bunch fully compresses (M = 0) for each sigma_z0."""
    return -sigma_z0 / spread if convention == "z" else sigma_z0 / spread


if __name__ == "__main__":
    # ------------------------- user settings -------------------------
    distribution = "parabola"   # "gaussian" or "parabola"
    convention = "z"            # "z" or "tau"
    color_by = "current"        # "current" or "sigma_z"
    FIGS = ps.output_dir(Path(__file__).resolve().parents[1], "tools", "figures")

    # One panel per operating point. Every beam quantity is free to edit:
    #   charge  bunch charge [C]
    #   spread  correlated dp/p0, SIGNED (its sign fixes the sign of R56)
    #   marker  working point to evaluate: (sigma_z,b BEFORE compression [m],
    #           R56 [m])
    # Seeded with the values measured from data/OP*_50k.dist.
    panels = [
        dict(name="OP1", branch_label="SASE 1 THz, dogleg",
             charge=1.000e-9, spread=+0.01,
             sigma_z0_axis=np.linspace(0.7e-3, 1.5e-3, 300),
             r56_axis=np.linspace(-0.15, -0.02, 300),
             current_cap=600.0, marker=(1.31e-3, -0.06)),
        dict(name="OP2", branch_label="SASE 10 THz, dogleg",
             charge=1.000e-9, spread=+0.0074,
             sigma_z0_axis=np.linspace(0.7e-3, 1.5e-3, 300),
             r56_axis=np.linspace(-0.15, -0.02, 300),
             current_cap=600.0, marker=(1.09e-3, -0.06)),
        dict(name="OP3", branch_label="SR 1 THz, chicane",
             charge=0.200e-9, spread=-0.006,
             sigma_z0_axis=np.linspace(0.5e-3, 1.1e-3, 300),
             r56_axis=np.linspace(0.05, 0.28, 300),
             current_cap=1200.0, marker=(0.76e-3, 0.1183)),
        dict(name="OP4", branch_label="SR 0.3 THz, chicane",
             charge=0.600e-9, spread=-0.0046,
             sigma_z0_axis=np.linspace(0.7e-3, 1.4e-3, 300),
             r56_axis=np.linspace(0.08, 0.32, 300),
             current_cap=1200.0, marker=(1.05e-3, 0.1955)),
    ]
    # -----------------------------------------------------------------

    # ---------------------------- compute ----------------------------
    profile = PROFILES[distribution]
    for panel in panels:
        spread, charge = panel["spread"], panel["charge"]
        sigma_z0, r56 = np.meshgrid(panel["sigma_z0_axis"], panel["r56_axis"])
        m_sz0, m_r56 = panel["marker"]
        sz_after = compressed_sigma_z(sigma_z0, r56, spread, convention)
        m_sz = compressed_sigma_z(m_sz0, m_r56, spread, convention)
        panel.update(
            sigma_z0=sigma_z0, r56=r56,
            sigma_z_after=sz_after,
            current_after=peak_current(charge, sz_after, profile),
            r56_full=full_compression_r56(m_sz0, spread, convention),
            marker_sigma_z=m_sz,
            marker_current=peak_current(charge, m_sz, profile))

    print(f"{distribution} profile, {convention} convention, dp/p0 fixed per OP\n")
    print(f"{'OP':>4s} {'Q/nC':>6s} {'dp/p0 %':>9s} {'sig_z bef/mm':>13s} "
          f"{'R56/mm':>9s} {'sig_z aft/mm':>13s} {'I_peak aft/A':>13s} "
          f"{'branch':>7s} {'full-comp R56/mm':>17s}")
    for panel in panels:
        m_sz0, m_r56 = panel["marker"]
        branch = "under" if abs(m_r56) < abs(panel["r56_full"]) else "over"
        print(f"{panel['name']:>4s} {panel['charge']*1e9:6.2f} "
              f"{panel['spread']*100:+9.4f} "
              f"{m_sz0*1e3:13.4f} {m_r56*1e3:+9.1f} "
              f"{panel['marker_sigma_z']*1e3:13.4f} "
              f"{panel['marker_current']:13.1f} {branch:>7s} "
              f"{panel['r56_full']*1e3:+17.1f}")

    # ----------------------------- plot ------------------------------
    ps.apply_style()
    fig, axes = plt.subplots(figsize=(8, 5.5), nrows=2, ncols=2,
                             layout="constrained")

    for ax, panel in zip(axes.flat, panels):
        x, y = panel["sigma_z0"] * 1e3, panel["r56"]
        m_sz0, m_r56 = panel["marker"]

        if color_by == "current":
            cf = ax.contourf(x, y, panel["current_after"],
                             levels=np.linspace(0.0, panel["current_cap"], 21),
                             cmap="YlGn", extend="max")
            cbar_label = r"$I_{\rm peak}$ after compression [$A$]"
        else:
            cf = ax.contourf(x, y, panel["sigma_z_after"] * 1e3,
                             levels=20, cmap="OrRd")
            cbar_label = r"$\sigma_{z,b}$ after compression [$mm$]"
        fig.colorbar(cf, ax=ax).set_label(cbar_label)

        # full-compression ridge (M = 0): the model's singular line
        ax.plot(panel["sigma_z0_axis"] * 1e3,
                full_compression_r56(panel["sigma_z0_axis"], panel["spread"],
                                     convention),
                color="#00e5ff", linewidth=1.8, linestyle="--",
                label="full compression")

        ax.plot(m_sz0 * 1e3, m_r56, "o", color="white", markersize=10,
                markeredgecolor="black", markeredgewidth=1.4, zorder=5,
                label=(rf"$R_{{56}}$ = {m_r56*1e3:+.1f} mm $\rightarrow$ "
                       rf"{panel['marker_current']:.0f} A" "\n"
                       rf"$\sigma_{{z,b}}$: {m_sz0*1e3:.3f} $\rightarrow$ "
                       rf"{panel['marker_sigma_z']*1e3:.3f} mm"))

        ax.set_ylim(panel["r56_axis"].min(), panel["r56_axis"].max())
        ax.set_title(f"{panel['name']} — {panel['branch_label']}\n"
                     rf"$Q$ = {panel['charge']*1e9:.1f} nC, "
                     rf"$\sigma_{{p_z}}/p_{{z0}}$ = "
                     rf"{panel['spread']*100:+.3f} %", fontsize=11)
        ax.legend(loc="lower left", framealpha=0.85, fontsize=9)

    for ax in axes.flat:
        ax.set_xlabel(r"$\sigma_{z,b}$ before compression [$mm$]")
        ax.set_ylabel(r"$R_{56}$ [$m$]")

    plt.show()
    ps.save(fig, FIGS, 'fig_linear_bc_yield')
