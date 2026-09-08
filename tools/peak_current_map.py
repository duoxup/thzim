#!/usr/bin/env python3
r"""Peak current from bunch charge and rms length, for a chosen current profile.

The one relation the whole compression chain is read against. Charge is
conserved, so a compressor moves a beam HORIZONTALLY across this map, at fixed
Q, until it lands on the current the FEL asks for. It is the same
peak_current() the bunch_compression_* tools use, drawn on its own so the
trade-off can be read without committing to a compressor.

Grid axes: rms bunch length sigma_z,b  x  bunch charge Q.
Contours:  peak current I_peak.

Both profile models make I_peak strictly proportional to Q and inversely
proportional to sigma_z,

    I_peak = k Q c / sigma_z,     k = 1/sqrt(2 pi) = 0.3989   (Gaussian)
                                  k = 3/(4 sqrt 5) = 0.3354   (inverted parabola)

so an iso-current line is Q = (I_peak/k c) sigma_z -- a straight line through
the origin on linear axes, a 45-degree line on log-log ones. Only the constant
separates the two shapes: at the same Q and sigma_z an inverted parabola peaks
at 0.841 x the Gaussian value, which is the level of accuracy this whole family
of estimates carries.

Note sigma_z,b is the RMS length, not the FWHM; profile.fwhm_over_sigma
converts (2.3548 Gaussian, 3.1623 parabola). The top axis carries that
conversion, reading the same length as the FWHM duration L_FWHM =
fwhm_over_sigma sigma_z / c -- so it shifts by 34 % between the two profiles.

Every input is a free parameter in the settings block. Markers are optional and
seeded with the four operating points AFTER compression; giving one a
`sigma_z_before` draws the compressor as the horizontal arrow it is.

Layout: physics is imported from bunch_compression_scan.py (same directory, so a
plain import works); this file is settings -> compute -> plot.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from scipy.constants import c as c_light

import plt_style as ps
from bunch_compression_scan import PROFILES, peak_current


if __name__ == "__main__":
    # ------------------------- user settings -------------------------
    distribution = "parabola"   # "gaussian" or "parabola"
    log_axes = True             # log-log (iso-current lines at 45 deg) or linear
    FIGS = ps.output_dir(Path(__file__).resolve().parents[1], "tools", "figures")

    # (min, max, n_points); the axes are sampled geometrically when log_axes
    sigma_z_range = (50e-6, 1000e-6, 300)   # rms bunch length [m]
    charge_range = (600e-12, 1500e-12, 300)    # bunch charge [C]

    # labelled iso-current lines [A]; those off the map are reported, not drawn
    i_levels = [10, 30, 100, 200, 400, 1000, 3000]     # None: no iso-lines

    # optional working points: name, charge [C], rms length [m]. `sigma_z_before`
    # adds the compressor arrow (charge is conserved, so it is horizontal);
    # `text_offset` nudges the label [points]. Set markers = [] to drop them.
    markers = [
        dict(name="OP1", charge=1.000e-9, sigma_z=0.5028e-3,
             sigma_z_before=1.31e-3, text_offset=(11, 6)),
        dict(name="OP2", charge=1.000e-9, sigma_z=0.5028e-3,
             sigma_z_before=1.09e-3, text_offset=(11, -16)),
        dict(name="OP3", charge=0.200e-9, sigma_z=0.0503e-3,
             sigma_z_before=0.76e-3, text_offset=(11, 6)),
        dict(name="OP4", charge=0.600e-9, sigma_z=0.1508e-3,
             sigma_z_before=1.05e-3, text_offset=(11, -16)),
    ]                                                   # None: no markers
    # -----------------------------------------------------------------

    # ---------------------------- compute ----------------------------
    profile = PROFILES[distribution]
    sample = np.geomspace if log_axes else np.linspace
    sigma_z_axis = sample(*sigma_z_range[:2], sigma_z_range[2])
    charge_axis = sample(*charge_range[:2], charge_range[2])

    sigma_z, charge = np.meshgrid(sigma_z_axis, charge_axis)
    i_peak = peak_current(charge, sigma_z, profile)

    k = profile.sigma_z_at(1.0, 1.0) / c_light      # I_peak = k Q c / sigma_z
    print(f"{profile.label}: I_peak = {k:.4f} Q c / sigma_z, "
          f"FWHM = {profile.fwhm_over_sigma:.4f} sigma_z")
    print(f"map spans {i_peak.min():.1f} .. {i_peak.max():.0f} A over "
          f"sigma_z = {sigma_z_axis.min()*1e3:.3f} .. {sigma_z_axis.max()*1e3:.3f} mm, "
          f"Q = {charge_axis.min()*1e9:.2f} .. {charge_axis.max()*1e9:.2f} nC\n")

    if markers:
        print(f"{'point':>6s} {'Q/nC':>6s} {'sig_z/mm':>9s} {'I_peak/A':>9s} "
              f"{'sig_z bef/mm':>13s} {'I_pk bef/A':>11s} {'C':>7s}")
        for m in markers:
            m["i_peak"] = peak_current(m["charge"], m["sigma_z"], profile)
            before = m.get("sigma_z_before")
            if before is None:
                print(f"{m['name']:>6s} {m['charge']*1e9:6.2f} "
                      f"{m['sigma_z']*1e3:9.4f} {m['i_peak']:9.1f} "
                      f"{'-':>13s} {'-':>11s} {'-':>7s}")
                continue
            m["i_peak_before"] = peak_current(m["charge"], before, profile)
            print(f"{m['name']:>6s} {m['charge']*1e9:6.2f} "
                  f"{m['sigma_z']*1e3:9.4f} {m['i_peak']:9.1f} "
                  f"{before*1e3:13.4f} {m['i_peak_before']:11.1f} "
                  f"{before/m['sigma_z']:7.2f}")

    # ----------------------------- plot ------------------------------
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4), layout="constrained")

    x, y = sigma_z * 1e3, charge * 1e9
    cf = ax.contourf(x, y, i_peak, levels=np.linspace(i_peak.min(),
                                                      i_peak.max(), 500),
                     norm=LogNorm(), 
                     cmap="magma")
    cbar = fig.colorbar(cf, ax=ax,
                        ticks=[70, 200, 500, 1000, 2000, 3000]
                        )
    cbar.set_label(r"$I_{\rm peak}$ [$A$]")
    # plain numbers rather than the default 10^n; ticks outside the range are
    # dropped by the colorbar itself, so this stays correct if the axes change
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    if i_levels:
        levels_in = [i for i in i_levels if i_peak.min() < i < i_peak.max()]
        for i in sorted(set(i_levels) - set(levels_in)):
            print(f"  [warn] I_peak = {i:g} A is off the map "
                  f"[{i_peak.min():.1f}, {i_peak.max():.0f}] A")
        if levels_in:
            cs = ax.contour(x, y, i_peak, levels=levels_in, colors="white",
                            linewidths=1.0, linestyles="-")
            ax.clabel(cs, fmt={i: f"{i:g} A" for i in levels_in}, fontsize=8)

    if markers:
        for m in markers:
            before = m.get("sigma_z_before")
            if before is not None:
                ax.annotate("", xy=(m["sigma_z"] * 1e3, m["charge"] * 1e9),
                            xytext=(before * 1e3, m["charge"] * 1e9),
                            arrowprops=dict(arrowstyle="->", color="#00e5ff",
                                            linewidth=1.4, shrinkA=0, shrinkB=3))
                ax.plot(before * 1e3, m["charge"] * 1e9, "o", color="#00e5ff",
                        markersize=4, markeredgecolor="black", markeredgewidth=0.6,
                        zorder=5)
            ax.plot(m["sigma_z"] * 1e3, m["charge"] * 1e9, "o", color="white",
                    markersize=7, markeredgecolor="black", markeredgewidth=1.2,
                    zorder=6)
            ax.annotate(m["name"], (m["sigma_z"] * 1e3, m["charge"] * 1e9),
                        textcoords="offset points", xytext=m.get("text_offset", (8, 8)),
                        fontsize=9, color="white", zorder=7)

    if log_axes:
        ax.set_xscale("linear")
        ax.set_yscale("linear")
    ax.set_xlim(sigma_z_axis.min() * 1e3, sigma_z_axis.max() * 1e3)
    ax.set_ylim(charge_axis.min() * 1e9, charge_axis.max() * 1e9)
    ax.set_xlabel(r"$\sigma_{z,b}$ [$mm$]")
    ax.set_ylabel(r"$Q$ [$nC$]")

    # top axis: the same length read as a FWHM duration. Pure rescaling, so it
    # tracks the primary axis on either scale -- but the factor is the profile's
    # own fwhm_over_sigma, i.e. this axis moves when `distribution` changes.
    mm_to_ps = profile.fwhm_over_sigma * 1e9 / c_light   # sigma_z [mm] -> [ps]
    secax = ax.secondary_xaxis("top", functions=(lambda v: v * mm_to_ps,
                                                 lambda v: v / mm_to_ps))
    secax.set_xlabel(r"$L_b$ [$ps$]")
    ax.set_title(f"Peak current — {profile.label} current profile", fontsize=11)

    plt.show()
    ps.save(fig, FIGS, 'fig_peak_current_map')
