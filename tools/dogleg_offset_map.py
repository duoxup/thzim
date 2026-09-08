#!/usr/bin/env python3
r"""Dogleg transverse offset: Delta_x over (bend angle, bend radius).

The secondary constraint that goes with dogleg_geometry_map.py. There, R56 =
-2 rho (theta - sin theta) fixed the dipoles; here the question is where that
puts the downstream beamline, since a dogleg exists precisely to translate the
axis sideways. It is deliberately a SEPARATE glance rather than an overlay:
Delta_x does not feed back into R56 at all.

Grid axes: bend angle theta  x  bend radius rho -- the same axes as
           dogleg_geometry_map.py, so the two read side by side.
Colour map: Delta_x = 2 rho (1 - cos theta) + L_gap sin theta.

Unlike R56, Delta_x is NOT a function of (theta, rho) alone: the straight
between the dipoles carries the second term. l_gap_ref is therefore a setting,
and the honest way to read the map is

    l_gap_ref = 0     the floor -- the offset the two dipoles impose on their
                      own, which no choice of straight can undo
    l_gap_ref > 0     the offset actually delivered by that layout

Since the offset is wanted rather than merely tolerated here, the useful
inversion runs the other way: pick Delta_x from the beamline layout and let the
straight close it, L_gap = (Delta_x - 2 rho (1 - cos theta)) / sin theta. That
is gap_for_offset(), printed per design point below and requiring only that the
floor stays under the target.

Geometry is imported from dogleg_geometry_map.py (same directory, so a plain
import works); this file is settings -> compute -> plot.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import plt_style as ps
from dogleg_geometry_map import (dipole_offset, gap_for_offset, offset,
                                 rho_for_r56, z_footprint)


if __name__ == "__main__":
    # ------------------------- user settings -------------------------
    FIGS = ps.output_dir(Path(__file__).resolve().parents[1], "tools", "figures")

    theta_range = (20.0, 60.0, 300)     # bend angle [deg]
    rho_range = (0.20, 1.50, 300)       # bend radius [m]

    # inter-dipole straight [m] the map is drawn at; 0 gives the dipole-only floor
    l_gap_ref = 2.72

    # labelled iso-Delta_x lines [m]; those off the map are reported, not drawn
    offset_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    # design points, solved from their R56 target: name, R56 [m], theta [deg]
    targets = [
        dict(name="OP1", r56=-0.060, theta_deg=40.0, text_offset=(10, 6)),
        dict(name="OP2", r56=-0.060, theta_deg=40.0, text_offset=(10, -18)),
    ]
    delta_x_design = 2.0                # offset the layout actually wants [m]
    # -----------------------------------------------------------------

    # ---------------------------- compute ----------------------------
    theta_axis = np.radians(np.linspace(*theta_range[:2], theta_range[2]))
    rho_axis = np.linspace(*rho_range[:2], rho_range[2])
    theta, rho = np.meshgrid(theta_axis, rho_axis)

    dx = offset(theta, rho, l_gap_ref)
    floor = dipole_offset(theta, rho)

    print(f"dogleg, Delta_x at L_gap = {l_gap_ref:.2f} m")
    print(f"map spans Delta_x = {dx.min():.2f} .. {dx.max():.2f} m "
          f"(dipole-only floor {floor.min():.3f} .. {floor.max():.3f} m)\n")

    for t in targets:
        th = np.radians(t["theta_deg"])
        rho_t = rho_for_r56(t["r56"], th)
        gap = gap_for_offset(delta_x_design, th, rho_t)
        t.update(theta=th, rho=rho_t, offset=offset(th, rho_t, l_gap_ref),
                 floor=dipole_offset(th, rho_t), gap=gap,
                 z_len=z_footprint(th, rho_t, gap))
    if targets:
        print(f"{'point':>6s} {'R56/mm':>8s} {'theta/deg':>10s} {'rho/m':>7s} "
              f"{'floor/m':>8s} {'Dx@ref/m':>9s} {'L_gap/m':>8s} "
              f"{'z-length/m':>11s}   (target Delta_x = {delta_x_design:.2f} m)")
        for t in targets:
            print(f"{t['name']:>6s} {t['r56']*1e3:8.1f} {t['theta_deg']:10.2f} "
                  f"{t['rho']:7.4f} {t['floor']:8.4f} {t['offset']:9.4f} "
                  f"{t['gap']:8.4f} {t['z_len']:11.3f}")
            if t["floor"] > delta_x_design:
                print(f"  [warn] {t['name']}: dipole-only floor "
                      f"{t['floor']:.3f} m already exceeds the target offset "
                      f"-- no straight can close it")

    # ----------------------------- plot ------------------------------
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4), layout="constrained")

    x, y = np.degrees(theta), rho
    cf = ax.contourf(x, y, dx, levels=40, cmap="cividis")
    fig.colorbar(cf, ax=ax).set_label(r"$\Delta x$ [$m$]")

    levels_in = [v for v in offset_levels if dx.min() < v < dx.max()]
    for v in sorted(set(offset_levels) - set(levels_in)):
        print(f"  [warn] Delta_x = {v:g} m is off the map "
              f"[{dx.min():.2f}, {dx.max():.2f}] m")
    if levels_in:
        cs = ax.contour(x, y, dx, levels=levels_in, colors="white",
                        linewidths=1.0)
        ax.clabel(cs, fmt={v: f"{v:g} m" for v in levels_in}, fontsize=8)

    for t in targets:
        ax.plot(t["theta_deg"], t["rho"], "o", color="white", markersize=7,
                markeredgecolor="black", markeredgewidth=1.2, zorder=6)
        ax.annotate(rf"{t['name']}  $\Delta x$ = {t['offset']:.2f} m",
                    (t["theta_deg"], t["rho"]), textcoords="offset points",
                    xytext=t.get("text_offset", (10, 6)), fontsize=8,
                    color="white", zorder=7)

    ax.set_xlim(np.degrees(theta_axis.min()), np.degrees(theta_axis.max()))
    ax.set_ylim(rho_axis.min(), rho_axis.max())
    ax.set_xlabel(r"$\theta$ [$\degree$]")
    ax.set_ylabel(r"$\rho$ [$m$]")
    ax.set_title(rf"Dogleg $\Delta x$ — $L_{{\rm gap}}$ = {l_gap_ref:.2f} m",
                 fontsize=11)

    plt.show()
    ps.save(fig, FIGS, 'fig_dogleg_offset_map')
