#!/usr/bin/env python3
r"""Dogleg sizing map: R56 over (bend angle, bend radius).

The dogleg counterpart to chicane_geometry_map.py. bunch_compression_scan.py
says how much R56 the beam needs; this says which two-dipole achromat delivers
it.

Grid axes: bend angle theta  x  bend radius rho.
Colour map: R56 (z convention), shown SIGNED -- it is negative for a dogleg,
because the SASE branch runs a positive chirp, which needs R56 < 0. The colour
scale is logarithmic in magnitude but keeps the sign, so it runs from the
strongest (most negative) R56 at the dark end to the weakest at the light end.

For the mirror-symmetric achromat

    START  B1(+t)  d1 QO d2 QI d3 | C | d3 QI d2 QO d1  B2(-t)  END

the achromat theorem fixes R56 from the dipoles alone -- the quadrupoles barely
enter, and the inter-dipole straight does not enter at all:

    R56_z = -2 rho (theta - sin theta)

so, dropping the velocity term, rho follows in CLOSED FORM,

    rho = |R56_z| / (2 (theta - sin theta))

with no root find. That is the whole reason this map is two-dimensional: R56 is
a function of (theta, rho) only. The transverse offset is a separate, decoupled
condition, closed afterwards by the straight between the dipoles,

    Delta_x = 2 rho (1 - cos theta) + L_gap sin theta

so it is neither an axis nor an overlay here. It is a secondary constraint and
gets its own map in dogleg_offset_map.py, which imports the geometry from this
file.

The velocity term -L_tot/(beta gamma)^2 is dropped on purpose. Checked against
the exact Ocelot map at the design point (theta = 40 deg, rho = 0.542 m), it is
worth 6.0 % at 15.4 MeV and 0.9 % at 39.4 MeV -- accepted, because the dogleg
only needs an approximate R56: its peak current is not critical and is trimmed
afterwards with the injector chirp.

Layout: geometry -> ``__main__`` (user settings, compute, plot). The geometry
functions are module level, so they import cleanly into src/thzim/dogleg.py when
that lands. Figures are shown, not saved by default.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm
from matplotlib.ticker import FuncFormatter
from scipy.constants import c as c_light, m_e, e as q_e

import plt_style as ps

M_E_GEV = m_e * c_light**2 / q_e * 1e-9        # electron rest energy [GeV]


# ------------------------------- geometry -------------------------------

def brho(energy_gev):
    """Magnetic rigidity B rho [T m] from the TOTAL energy [GeV]."""
    return np.sqrt(energy_gev**2 - M_E_GEV**2) * 1e9 / c_light


def r56_z(theta, rho):
    """R56 [m], z convention (< 0 for a dogleg), geometry only."""
    return -2.0 * rho * (theta - np.sin(theta))


def rho_for_r56(r56_z_target, theta):
    """Bend radius [m] delivering the requested R56 -- the closed-form inverse."""
    return np.abs(r56_z_target) / (2.0 * (theta - np.sin(theta)))


def dipole_offset(theta, rho):
    """Transverse offset [m] from the two dipoles alone (L_gap = 0)."""
    return 2.0 * rho * (1.0 - np.cos(theta))


def offset(theta, rho, l_gap):
    """Total transverse offset Delta_x [m], closed by the inter-dipole straight."""
    return dipole_offset(theta, rho) + l_gap * np.sin(theta)


def gap_for_offset(delta_x, theta, rho):
    """Straight length [m] between the dipoles that closes Delta_x."""
    return (delta_x - dipole_offset(theta, rho)) / np.sin(theta)


def arc_length(theta, rho):
    """Dipole arc length [m]."""
    return rho * theta


def z_footprint(theta, rho, l_gap):
    """Longitudinal extent [m] of the two dipoles plus the straight."""
    return 2.0 * rho * np.sin(theta) + l_gap * np.cos(theta)


def field(rho, energy_gev):
    """Dipole field [T] at the given total energy."""
    return brho(energy_gev) / rho


if __name__ == "__main__":
    # ------------------------- user settings -------------------------
    FIGS = ps.output_dir(Path(__file__).resolve().parents[1], "tools", "figures")

    theta_range = (20.0, 60.0, 300)     # bend angle [deg]
    rho_range = (0.20, 1.50, 300)       # bend radius [m]

    # labelled iso-R56 lines [mm], signed; those off the map are reported
    r56_levels = [-400, -200, -100, -60, -50, -25, -10, -5]

    # design points to solve and mark: name, R56 target [m], bend angle [deg]
    targets = [
        dict(name="OP1 & OP2", r56=-0.060, theta_deg=40.0, text_offset=(10, 6)),
        # dict(name="OP2", r56=-0.060, theta_deg=40.0, text_offset=(10, -18)),
    ]
    delta_x_design = 2.0                # for the L_gap / footprint columns [m]
    field_energy_mev = [15.4, 39.4]     # kinetic energies for the B column; [] to skip
    # -----------------------------------------------------------------

    # ---------------------------- compute ----------------------------
    theta_axis = np.radians(np.linspace(*theta_range[:2], theta_range[2]))
    rho_axis = np.linspace(*rho_range[:2], rho_range[2])
    theta, rho = np.meshgrid(theta_axis, rho_axis)

    r56 = r56_z(theta, rho)

    print("dogleg, geometry-only R56 (no velocity term)")
    print(f"map spans R56 = {r56.min()*1e3:.0f} .. {r56.max()*1e3:.1f} mm\n")

    for t in targets:
        th = np.radians(t["theta_deg"])
        rho_t = rho_for_r56(t["r56"], th)
        gap = gap_for_offset(delta_x_design, th, rho_t)
        t.update(theta=th, rho=rho_t, arc=arc_length(th, rho_t), gap=gap,
                 z_len=z_footprint(th, rho_t, gap))
    if targets:
        print(f"{'point':>6s} {'R56/mm':>8s} {'theta/deg':>10s} {'rho/m':>7s} "
              f"{'L_B,arc/m':>10s} {'L_gap/m':>8s} {'z-length/m':>11s}"
              f"   (Delta_x = {delta_x_design:.2f} m)")
        for t in targets:
            print(f"{t['name']:>6s} {t['r56']*1e3:8.1f} {t['theta_deg']:10.2f} "
                  f"{t['rho']:7.4f} {t['arc']:10.4f} {t['gap']:8.4f} "
                  f"{t['z_len']:11.3f}")
        for ekin in field_energy_mev:
            e_tot = (ekin + M_E_GEV * 1e3) * 1e-3
            print(f"  B at E_kin = {ekin:5.1f} MeV: "
                  + ", ".join(f"{t['name']} {field(t['rho'], e_tot):.4f} T"
                              for t in targets))

    # ----------------------------- plot ------------------------------
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4), layout="constrained")

    x, y = np.degrees(theta), rho
    r56_mm = r56 * 1e3
    # R56 is single-signed here, so a log scale still reads well -- carry the
    # sign through the norm instead of plotting |R56|
    cf = ax.contourf(x, y, r56_mm,
                     levels=-np.geomspace(-r56_mm.min(), -r56_mm.max(), 200),
                     norm=FuncNorm((lambda v: -np.log10(-v),
                                    lambda u: -10.0**(-u)),
                                   vmin=r56_mm.min(), vmax=r56_mm.max()),
                     cmap="magma_r")
    cbar = fig.colorbar(cf, ax=ax,
                        ticks=[-1000, -300, -100, -30, -10, -3, -1])
    cbar.set_label(r"$R_{56}$ [$mm$]")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    levels_in = [v for v in r56_levels if r56_mm.min() < v < r56_mm.max()]
    for v in sorted(set(r56_levels) - set(levels_in)):
        print(f"  [warn] R56 = {v:g} mm is off the map "
              f"[{r56_mm.min():.0f}, {r56_mm.max():.1f}] mm")
    if levels_in:
        # levels are negative, and matplotlib dashes those by default
        cs = ax.contour(x, y, r56_mm, levels=levels_in, colors="white",
                        linewidths=1.0, linestyles="-")
        ax.clabel(cs, fmt={v: f"{v:g} mm" for v in levels_in}, fontsize=8)

    for t in targets:
        ax.plot(t["theta_deg"], t["rho"], "o", color="white", markersize=7,
                markeredgecolor="black", markeredgewidth=1.2, zorder=6)
        ax.annotate(
            # rf"{t['name']}  $\rho$ = {t['rho']:.3f} m",
            rf"{t['name']}",
            (t["theta_deg"], t["rho"]), textcoords="offset points",
            xytext=t.get("text_offset", (10, 6)), fontsize=8,
            color="white", zorder=7)

    ax.set_xlim(np.degrees(theta_axis.min()), np.degrees(theta_axis.max()))
    ax.set_ylim(rho_axis.min(), rho_axis.max())
    ax.set_xlabel(r"$\theta$ [$\degree$]")
    ax.set_ylabel(r"$\rho$ [$m$]")
    ax.set_title(r"Dogleg $R_{56} = -2\rho\,(\theta - \sin\theta)$",
                 fontsize=11)

    plt.show()
    ps.save(fig, FIGS, 'fig_dogleg_geometry_map')
