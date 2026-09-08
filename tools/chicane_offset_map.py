#!/usr/bin/env python3
r"""Chicane transverse excursion: Delta_x over (bend angle, drift z-footprint).

The secondary constraint that goes with chicane_geometry_map.py. R56 sets the
compression; Delta_x sets how far off axis the beam swings at the middle of the
chicane, which is what the vacuum chamber and the dipole gap have to accept. It
is deliberately a SEPARATE glance rather than an overlay: nothing here changes
the R56 choice, it only says whether that choice fits in the hardware.

Grid axes: bend angle theta  x  dipole-to-dipole drift z-projection L_Dz
           -- the same axes as chicane_geometry_map.py, so the two read
           side by side.
Colour map: Delta_x = 2 L_Bz tan(theta/2) + L_Dz tan(theta).

Both terms grow with theta, the second one also with L_Dz, so Delta_x rises
towards the top-right exactly where R56 does -- which is the whole tension of
the design: more compression costs aperture. The design points are solved from
their R56 targets (same call as the R56 map) and marked, so the readout is
"the theta my R56 needs puts the beam this far off axis".

Geometry is imported from chicane_geometry_map.py (same directory, so a plain
import works); this file is settings -> compute -> plot.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import plt_style as ps
from chicane_geometry_map import (arc_length, bend_radius, offset,
                                  theta_for_r56)


if __name__ == "__main__":
    # ------------------------- user settings -------------------------
    l_bz = 0.10                 # dipole z-projection [m], FIXED
    FIGS = ps.output_dir(Path(__file__).resolve().parents[1], "tools", "figures")

    theta_range = (5.0, 25.0, 300)      # bend angle [deg]
    l_dz_range = (0.30, 1.50, 300)      # drift z-projection [m]

    # labelled iso-Delta_x lines [mm]; those off the map are reported, not drawn
    offset_levels = [50, 100, 150, 200, 300, 400, 500, 600]

    # design points, solved from their R56 target: name, R56 [m], L_Dz [m]
    targets = [
        dict(name="OP3", r56=+0.1467, l_dz=0.75, text_offset=(9, 6)),
        dict(name="OP4", r56=+0.1953, l_dz=0.75, text_offset=(9, -18)),
    ]
    # -----------------------------------------------------------------

    # ---------------------------- compute ----------------------------
    theta_axis = np.radians(np.linspace(*theta_range[:2], theta_range[2]))
    l_dz_axis = np.linspace(*l_dz_range[:2], l_dz_range[2])
    theta, l_dz = np.meshgrid(theta_axis, l_dz_axis)

    dx = offset(theta, l_bz, l_dz)

    print(f"chicane, L_Bz = {l_bz*1e3:.0f} mm fixed")
    print(f"map spans Delta_x = {dx.min()*1e3:.0f} .. {dx.max()*1e3:.0f} mm\n")

    for t in targets:
        th = theta_for_r56(t["r56"], l_bz, t["l_dz"])
        t.update(theta=th, offset=offset(th, l_bz, t["l_dz"]),
                 rho=bend_radius(th, l_bz), arc=arc_length(th, l_bz))
    if targets:
        print(f"{'point':>6s} {'R56/mm':>8s} {'L_Dz/m':>7s} {'theta/deg':>10s} "
              f"{'Dx/mm':>8s} {'rho/m':>7s}")
        for t in targets:
            print(f"{t['name']:>6s} {t['r56']*1e3:8.1f} {t['l_dz']:7.2f} "
                  f"{np.degrees(t['theta']):10.3f} {t['offset']*1e3:8.1f} "
                  f"{t['rho']:7.3f}")

    # ----------------------------- plot ------------------------------
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4), layout="constrained")

    x, y = np.degrees(theta), l_dz
    cf = ax.contourf(x, y, dx * 1e3, levels=40, cmap="cividis")
    fig.colorbar(cf, ax=ax).set_label(r"$\Delta x$ [$mm$]")

    levels_in = [v for v in offset_levels
                 if dx.min() * 1e3 < v < dx.max() * 1e3]
    for v in sorted(set(offset_levels) - set(levels_in)):
        print(f"  [warn] Delta_x = {v:g} mm is off the map "
              f"[{dx.min()*1e3:.0f}, {dx.max()*1e3:.0f}] mm")
    if levels_in:
        cs = ax.contour(x, y, dx * 1e3, levels=levels_in, colors="white",
                        linewidths=1.0)
        ax.clabel(cs, fmt={v: f"{v:g} mm" for v in levels_in}, fontsize=8)

    for t in targets:
        ax.plot(np.degrees(t["theta"]), t["l_dz"], "o", color="white",
                markersize=7, markeredgecolor="black", markeredgewidth=1.2,
                zorder=6)
        ax.annotate(rf"{t['name']}  $\Delta x$ = {t['offset']*1e3:.0f} mm",
                    (np.degrees(t["theta"]), t["l_dz"]),
                    textcoords="offset points",
                    xytext=t.get("text_offset", (9, 6)),
                    fontsize=8, color="white", zorder=7)

    ax.set_xlim(np.degrees(theta_axis.min()), np.degrees(theta_axis.max()))
    ax.set_ylim(l_dz_axis.min(), l_dz_axis.max())
    ax.set_xlabel(r"$\theta$ [$\degree$]")
    ax.set_ylabel(r"$L_{D,z}$ [$m$]")
    ax.set_title(rf"Chicane $\Delta x$ — $L_{{B,z}}$ = {l_bz*1e3:.0f} mm fixed",
                 fontsize=11)

    plt.show()
    ps.save(fig, FIGS, 'fig_chicane_offset_map')
