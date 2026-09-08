#!/usr/bin/env python3
r"""Bunch-compressor R56 map: the transpose of bunch_compression_scan.py.

Same linear model, same sizing question, axes swapped. ``bunch_compression_scan``
puts R56 on the y axis and colours by the correlated momentum spread the
injector must deliver; here the spread IS the y axis and the colour map is the
R56 that spread requires. Read it as "the injector gives me this chirp -- how
long does the compressor have to be?", which is the form the chicane/dogleg
footprint is actually sized from.

Grid axes: rms bunch length sigma_z,b BEFORE compression  x  correlated
           sigma_pz/pz0 available.
Colour map: the R56 required to reach the target current.

Both views collapse to the same one-liner. With M = sigma_final/sigma_initial
and the target length sigma_t = sigma_z,b(after) fixed by charge and target
current,

    z convention:    R56 = (M - 1) sigma_z0 / spread = (sigma_t - sigma_z0) / spread
    tau convention:  R56 = (1 - M) sigma_z0 / spread

for under-compression (M = +1/C); over-compression flips the sign of M, i.e.
the numerator becomes -(sigma_t + sigma_z0). So R56 goes as 1/spread: halving
the chirp doubles the compressor. Because the numerator carries the sign of
(sigma_t - sigma_z0) < 0, R56 takes the OPPOSITE sign to the spread -- a
positive chirp (SASE branch, OP1/OP2) needs R56 < 0, a dogleg; a negative chirp
(SR branch, OP3/OP4) needs R56 > 0, a chicane. The spread axis must therefore
stay on one side of zero; R56 diverges as it crosses.

The compression factor C = sigma_z0/sigma_t depends on the x axis alone, so its
contours are vertical -- they were in the scan view too, and they still give the
"how hard am I squeezing" readout at a glance.

Every beam input is a free parameter in the settings block -- charge, correlated
spread, bunch length before compression, target current -- seeded with the
values measured from data/OP*_50k.dist. Nothing here reads a distribution; to
refresh the seeds use measure_beam() from bunch_compression_scan.py.

Layout: physics is imported from bunch_compression_scan.py (same directory, so a
plain import works); this file is settings -> compute -> plot.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import plt_style as ps
from bunch_compression_scan import (PROFILES, beta_squared, peak_current,
                                    required_r56)


if __name__ == "__main__":
    # ------------------------- user settings -------------------------
    distribution = "parabola"   # "gaussian" or "parabola"
    convention = "z"            # "z" or "tau"  (sets the sign of h and dp/p0)
    branch = "under"            # "under" or "over" compression
    FIGS = ps.output_dir(Path(__file__).resolve().parents[1], "tools", "figures")

    # One panel per operating point. Every beam quantity is free to edit:
    #   charge          bunch charge [C]
    #   spread          correlated dp/p0 the injector delivers, SIGNED -- marks
    #                   the working point on the y axis
    #   sigma_z_before  rms bunch length before compression [m]
    #   peak_current    target peak current after compression [A]
    #   spread_axis     must not cross zero (R56 diverges there); its sign has
    #                   to match `spread`
    #   c_levels        compression-factor contours (vertical, see the module
    #                   docstring); c_label_frac places their labels at that
    #                   fraction of the y range, clear of legend_loc
    # The background uses the panel's own charge and target, so the iso-R56 line
    # passes through the solved marker by construction.
    panels = [
        dict(name="OP1", branch_label="SASE 1 THz, dogleg",
             charge=1.000e-9, spread=+0.01, sigma_z_before=1.31e-3,
             peak_current=200.0,
             sigma_z0_axis=np.linspace(0.7e-3, 1.5e-3, 200),
             spread_axis=np.linspace(0.006, 0.016, 200),
             c_levels=[2.0, 2.25, 2.5],
             legend_loc="upper left", c_label_frac=0.28),
        dict(name="OP2", branch_label="SASE 10 THz, dogleg",
             charge=1.000e-9, spread=+0.0074, sigma_z_before=1.09e-3,
             peak_current=200.0,
             sigma_z0_axis=np.linspace(0.7e-3, 1.5e-3, 200),
             spread_axis=np.linspace(0.005, 0.012, 200),
             c_levels=[1.5, 2.0, 2.5],
             legend_loc="upper left", c_label_frac=0.28),
        dict(name="OP3", branch_label="SR 1 THz, chicane",
             charge=0.200e-9, spread=-0.006, sigma_z_before=0.76e-3,
             peak_current=400.0,
             sigma_z0_axis=np.linspace(0.5e-3, 1.1e-3, 200),
             spread_axis=np.linspace(-0.010, -0.004, 200),
             c_levels=[10, 15, 20],
             legend_loc="lower left", c_label_frac=0.78),
        dict(name="OP4", branch_label="SR 0.3 THz, chicane",
             charge=0.600e-9, spread=-0.0046, sigma_z_before=1.05e-3,
             peak_current=400.0,
             sigma_z0_axis=np.linspace(0.7e-3, 1.4e-3, 200),
             spread_axis=np.linspace(-0.008, -0.003, 200),
             c_levels=[5, 7, 9],
             legend_loc="lower left", c_label_frac=0.78),
    ]
    ekin_ev = 15.4e6            # for the sigma_E/E0 conversion print; None = skip
    # -----------------------------------------------------------------

    # ---------------------------- compute ----------------------------
    profile = PROFILES[distribution]
    for panel in panels:
        sz_before, charge = panel["sigma_z_before"], panel["charge"]
        sz_target = profile.sigma_z_at(charge, panel["peak_current"])

        axis = panel["spread_axis"]
        if axis.min() * axis.max() <= 0.0:
            raise ValueError(f"{panel['name']}: spread_axis crosses zero, where "
                             f"R56 diverges -- keep it on one side")
        if axis.min() * panel["spread"] < 0.0:
            print(f"  [warn] {panel['name']}: spread {panel['spread']:+.4f} has "
                  f"the opposite sign to spread_axis -- the marker is off-panel")

        sigma_z0, spread = np.meshgrid(panel["sigma_z0_axis"], axis)
        panel.update(
            sigma_z_target=sz_target,
            compression=sz_before / sz_target,
            current_before=peak_current(charge, sz_before, profile),
            r56=required_r56(sz_before, panel["spread"], sz_target,
                             convention, branch),
            sigma_z0=sigma_z0, spread_grid=spread,
            compression_grid=sigma_z0 / sz_target,
            r56_grid=required_r56(sigma_z0, spread, sz_target,
                                  convention, branch))

    print(f"{distribution}, {branch}-compression, {convention} convention")
    if ekin_ev is not None:
        b2 = beta_squared(ekin_ev)
        print(f"sigma_E/E0 = {b2:.6f} x dp/p0 at E_kin = {ekin_ev/1e6:.1f} MeV "
              f"({(1-b2)*100:.2f} % lower)\n")
    print(f"{'OP':>4s} {'Q/nC':>6s} {'sig_z bef/mm':>13s} {'I_pk bef/A':>11s} "
          f"{'I_pk tgt/A':>11s} {'sig_z tgt/mm':>13s} {'C':>7s} "
          f"{'dp/p0 %':>9s} {'R56 req/mm':>11s}")
    for panel in panels:
        print(f"{panel['name']:>4s} {panel['charge']*1e9:6.2f} "
              f"{panel['sigma_z_before']*1e3:13.4f} "
              f"{panel['current_before']:11.1f} {panel['peak_current']:11.0f} "
              f"{panel['sigma_z_target']*1e3:13.4f} {panel['compression']:7.2f} "
              f"{panel['spread']*100:+9.4f} {panel['r56']*1e3:+11.1f}")

    # ----------------------------- plot ------------------------------
    ps.apply_style()
    fig, axes = plt.subplots(figsize=(8, 5.5), nrows=2, ncols=2,
                             layout="constrained")

    for ax, panel in zip(axes.flat, panels):
        x = panel["sigma_z0"] * 1e3
        y = panel["spread_grid"] * 100.0
        r56_mm = panel["r56_grid"] * 1e3
        level = panel["r56"] * 1e3

        cf = ax.contourf(x, y, r56_mm, levels=20, cmap="viridis")
        fig.colorbar(cf, ax=ax).set_label(r"Required $R_{56}$ [$mm$]")

        # C = sigma_z0/sigma_t depends on x alone, so each contour is the
        # vertical line x = C sigma_t -- label them by hand at a fixed height
        # rather than letting matplotlib drop them on top of the legend
        x_axis = panel["sigma_z0_axis"] * 1e3
        # c_at = {c: c * panel["sigma_z_target"] * 1e3 for c in panel["c_levels"]}
        # c_in = [c for c, xc in c_at.items() if x_axis.min() < xc < x_axis.max()]
        # if c_in:
        #     y_lab = (y.min() + panel["c_label_frac"] * (y.max() - y.min()))
        #     cs_c = ax.contour(x, y, panel["compression_grid"], levels=c_in,
        #                       colors="black", linewidths=1.1, linestyles="--")
        #     ax.clabel(cs_c, fmt={c: f"C = {c:g}" for c in c_in}, fontsize=9,
        #               manual=[(c_at[c], y_lab) for c in c_in])
        # for c in set(panel["c_levels"]) - set(c_in):
        #     print(f"  [warn] {panel['name']}: C = {c:g} sits at "
        #           f"{c_at[c]:.2f} mm, off the sigma_z0 axis")

        # # iso-line at the R56 this working point implies
        # if r56_mm.min() <= level <= r56_mm.max():
        #     cs_op = ax.contour(x, y, r56_mm, levels=[level], colors="white",
        #                        linewidths=1.8, linestyles="-")
        #     ax.clabel(cs_op, fmt={level: f"{level:+.1f} mm"}, fontsize=9)
        # else:
        #     print(f"  [warn] {panel['name']}: R56 {level:+.1f} mm outside the "
        #           f"panel range [{r56_mm.min():+.1f}, {r56_mm.max():+.1f}] mm "
        #           f"-- widen spread_axis or sigma_z0_axis")

        ax.plot(panel["sigma_z_before"] * 1e3, panel["spread"] * 100.0, "o",
                color="white", markersize=10, markeredgecolor="black",
                markeredgewidth=1.4, zorder=5,
                label=(rf"before: $\sigma_{{z,b}}$ = "
                       rf"{panel['sigma_z_before']*1e3:.2f} mm, "
                       rf"$C$ = {panel['compression']:.2f}" "\n"
                       rf"$R_{{56}}$ = {panel['r56']*1e3:+.1f} mm"))

        ax.set_title(f"{panel['name']} — {panel['branch_label']}\n"
                     rf"$Q$ = {panel['charge']*1e9:.1f} nC, "
                     rf"target $I_{{\rm peak}}$ = "
                     rf"{panel['peak_current']:.0f} A "
                     rf"($\sigma_{{z,b}}$ after = "
                     rf"{panel['sigma_z_target']*1e3:.2f} mm)", fontsize=11)
        # ax.legend(loc=panel["legend_loc"], framealpha=0.85, fontsize=9)

    for ax in axes.flat:
        ax.set_xlabel(r"$\sigma_{z,b}$ before compression [$mm$]")
        ax.set_ylabel(r"$\sigma_{pz}^\mathrm{cor}/p_{z0}$ [%]")

    plt.show()
    ps.save(fig, FIGS, 'fig_estimate_req_r56_map')
