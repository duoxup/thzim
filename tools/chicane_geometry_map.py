#!/usr/bin/env python3
r"""Chicane sizing map: R56 over (bend angle, drift z-footprint), L_Bz fixed.

The sizing-stage counterpart to bunch_compression_scan.py. That tool says how
much R56 the beam needs; this one says which four-dipole chicane delivers it.

Grid axes: bend angle theta  x  dipole-to-dipole drift z-projection L_Dz.
Colour map: R56 (z convention, positive for a chicane).

What is fixed on a real beamline is the LONGITUDINAL footprint of the hardware,
so the geometry is parameterised by z-projections and theta is the tuning knob:

    rho          = L_Bz / sin(theta)                      bend radius
    L_B,arc      = rho theta = L_Bz theta / sin(theta)     dipole arc
    L_D,path     = L_Dz / cos(theta)                       drift at angle theta
    Delta_x      = 2 L_Bz tan(theta/2) + L_Dz tan(theta)   mid-chicane offset
    B            = B rho sin(theta) / L_Bz                 dipole field

    R56_z ~ +2 theta^2 (L_D,path + 2/3 L_B,arc)

The R56 expression is the SMALL-ANGLE form; the velocity term -L_tot/(beta gamma)^2
is dropped on purpose. Checked against the exact Ocelot RBend map at the design
angles, the small-angle expansion reads 1.6 % low at 16.7 deg and 2.5 % low at
19.1 deg, while the dropped velocity term is worth only 0.2-0.6 % there. That is
deliberate: the chicane is built with margin and trimmed afterwards with theta,
so this map only has to place the footprint, not predict it.

Delta_x is reported per design point but deliberately NOT drawn here -- it is a
secondary constraint and gets its own map in chicane_offset_map.py, which
imports the geometry from this file.

Layout: geometry -> ``__main__`` (user settings, compute, plot). The geometry
functions are module level, so they import cleanly into src/thzim/chicane.py
when that lands. Figures are shown, not saved by default.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from scipy.constants import c as c_light, m_e, e as q_e
from scipy.optimize import brentq

import plt_style as ps

M_E_GEV = m_e * c_light**2 / q_e * 1e-9        # electron rest energy [GeV]


# ------------------------------- geometry -------------------------------

def brho(energy_gev):
    """Magnetic rigidity B rho [T m] from the TOTAL energy [GeV]."""
    return np.sqrt(energy_gev**2 - M_E_GEV**2) * 1e9 / c_light


def bend_radius(theta, l_bz):
    """rho [m] of a dipole whose arc projects to L_Bz along z."""
    return l_bz / np.sin(theta)


def arc_length(theta, l_bz):
    """Dipole arc length [m]; -> L_Bz as theta -> 0."""
    return l_bz * theta / np.sin(theta)


def drift_path(theta, l_dz):
    """Path length [m] of the inter-dipole drift, travelled at angle theta."""
    return l_dz / np.cos(theta)


def offset(theta, l_bz, l_dz):
    """Transverse excursion Delta_x [m] at the middle of the chicane."""
    return 2.0 * l_bz * np.tan(0.5 * theta) + l_dz * np.tan(theta)


def r56_z(theta, l_bz, l_dz):
    """R56 [m], z convention (> 0 for a chicane), small-angle, no velocity term."""
    return 2.0 * theta**2 * (drift_path(theta, l_dz)
                             + (2.0 / 3.0) * arc_length(theta, l_bz))


def field(theta, l_bz, energy_gev):
    """Dipole field [T] at the given total energy."""
    return brho(energy_gev) * np.sin(theta) / l_bz


def theta_for_r56(r56_target, l_bz, l_dz, bracket=(1e-3, 1.0)):
    """Bend angle [rad] giving the requested R56 at this footprint."""
    return brentq(lambda t: r56_z(t, l_bz, l_dz) - r56_target, *bracket,
                  xtol=1e-12)


if __name__ == "__main__":
    # ------------------------- user settings -------------------------
    l_bz = 0.10                 # dipole z-projection [m], FIXED
    FIGS = ps.output_dir(Path(__file__).resolve().parents[1], "tools", "figures")

    theta_range = (5.0, 25.0, 300)      # bend angle [deg]
    l_dz_range = (0.30, 1.50, 300)      # drift z-projection [m]

    # labelled iso-R56 lines [mm]; those off the map are reported, not drawn
    r56_levels = [25, 50, 100, 150, 200, 300, 500]

    # design points to solve and mark: name, R56 target [m], drift footprint [m]
    targets = [
        dict(name="OP3", r56=+0.1467, l_dz=0.75, text_offset=(9, 6)),
        dict(name="OP4", r56=+0.1953, l_dz=0.75, text_offset=(9, -16)),
    ]
    field_energy_mev = [39.9, 21.8]     # kinetic energies for the B column; [] to skip
    # -----------------------------------------------------------------

    # ---------------------------- compute ----------------------------
    theta_axis = np.radians(np.linspace(*theta_range[:2], theta_range[2]))
    l_dz_axis = np.linspace(*l_dz_range[:2], l_dz_range[2])
    theta, l_dz = np.meshgrid(theta_axis, l_dz_axis)

    r56 = r56_z(theta, l_bz, l_dz)

    print(f"chicane, L_Bz = {l_bz*1e3:.0f} mm fixed, small-angle R56, "
          f"no velocity term")
    print(f"map spans R56 = {r56.min()*1e3:.1f} .. {r56.max()*1e3:.0f} mm\n")

    for t in targets:
        th = theta_for_r56(t["r56"], l_bz, t["l_dz"])
        t.update(theta=th, rho=bend_radius(th, l_bz),
                 arc=arc_length(th, l_bz), offset=offset(th, l_bz, t["l_dz"]),
                 footprint=4 * l_bz + 2 * t["l_dz"])
    if targets:
        print(f"{'point':>6s} {'R56/mm':>8s} {'L_Dz/m':>7s} {'theta/deg':>10s} "
              f"{'rho/m':>7s} {'L_B,arc/m':>10s} {'Dx/mm':>7s} {'z-length/m':>11s}")
        for t in targets:
            print(f"{t['name']:>6s} {t['r56']*1e3:8.1f} {t['l_dz']:7.2f} "
                  f"{np.degrees(t['theta']):10.3f} {t['rho']:7.3f} "
                  f"{t['arc']:10.4f} {t['offset']*1e3:7.1f} "
                  f"{t['footprint']:11.2f}")
        for ekin in field_energy_mev:
            e_tot = (ekin + M_E_GEV * 1e3) * 1e-3
            fields = [field(t["theta"], l_bz, e_tot) for t in targets]
            print(f"  B at E_kin = {ekin:5.1f} MeV: "
                  + ", ".join(f"{t['name']} {b:.4f} T"
                              for t, b in zip(targets, fields)))

    # ----------------------------- plot ------------------------------
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4), layout="constrained")

    x, y = np.degrees(theta), l_dz
    cf = ax.contourf(x, y, r56 * 1e3,
                     levels=np.geomspace(r56.min() * 1e3, r56.max() * 1e3, 200),
                     norm=LogNorm(), cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax, ticks=[1, 3, 10, 30, 100, 300, 1000, 3000])
    cbar.set_label(r"$R_{56}$ [$mm$]")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    levels_in = [v for v in r56_levels if r56.min() * 1e3 < v < r56.max() * 1e3]
    for v in sorted(set(r56_levels) - set(levels_in)):
        print(f"  [warn] R56 = {v:g} mm is off the map "
              f"[{r56.min()*1e3:.1f}, {r56.max()*1e3:.0f}] mm")
    if levels_in:
        cs = ax.contour(x, y, r56 * 1e3, levels=levels_in, colors="white",
                        linewidths=1.0)
        ax.clabel(cs, fmt={v: f"{v:g} mm" for v in levels_in}, fontsize=8)

    for t in targets:
        ax.plot(np.degrees(t["theta"]), t["l_dz"], "o", color="white",
                markersize=7, markeredgecolor="black", markeredgewidth=1.2,
                zorder=6)
        ax.annotate(f"{t['name']}\n{np.degrees(t['theta']):.2f}°",
                    (np.degrees(t["theta"]), t["l_dz"]),
                    textcoords="offset points",
                    xytext=t.get("text_offset", (9, 6)),
                    fontsize=8, color="white", zorder=7)

    ax.set_xlim(np.degrees(theta_axis.min()), np.degrees(theta_axis.max()))
    ax.set_ylim(l_dz_axis.min(), l_dz_axis.max())
    ax.set_xlabel(r"$\theta$ [$\degree$]")
    ax.set_ylabel(r"$L_{D,z}$ [$m$]")
    ax.set_title(rf"Chicane $R_{{56}}$ — $L_{{B,z}}$ = {l_bz*1e3:.0f} mm fixed",
                 fontsize=11)

    plt.show()
    ps.save(fig, FIGS, 'fig_chicane_geometry_map')
