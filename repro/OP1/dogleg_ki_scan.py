#!/usr/bin/env python3
r"""OP1 dogleg — step 1 of 2: scan the inner-pair strength ki.

The dogleg geometry is already fixed (rho and theta come from the R56 target,
see tools/dogleg_geometry_map.py; Delta_x from the layout). What is left free is
the inner quad pair ki, which the achromat does not constrain. This script walks
ki and reports what each choice costs, so the next script can be run at the ki
you pick.

Per ki, two solves from ``thzim.dogleg``:

  API 1  solve_achromat_quads(geom, ki) -> ko
         one condition Dx(C) = 0 pins the outer pair, because mirror symmetry
         plus antisymmetric dipoles collapse the two exit conditions
         Dx = Dx' = 0 into that single one. The exit Dx / Dx' fall out of the
         same call and are the achromat's own check -- they do NOT depend on the
         entrance beta/alpha, since dispersion is driven by the dipoles with
         Dx = Dx' = 0 at the entrance.
  API 2  solve_entrance_twiss(dl) -> entrance Twiss + peak betas
         the peak betas are a BY-PRODUCT of the min-peak search, not a separate
         forward pass: _min_peak() already ran the full Twiss for every trial
         centre beta. dogleg_forward.py re-runs it only to draw and verify it.

Two panels:
  (a) peak beta per plane and the score being minimised, winner marked
  (b) ko and R56 against ki -- the orthogonality the layered design rests on:
      ki forces ko to retune, and leaves R56 alone

`emit_ratio` = eps_y/eps_x is the only place a beam property may enter, and it
only reweights the score in (a): peak beta is optical, but the size that has to
fit is sigma^2 = eps beta, so with unequal emittances the roundest-beta ki is
not the roundest-beam ki. Leave it at 1.0 to keep the design distribution-free.

Run:  python dogleg_ki_scan.py     (needs `pip install -e .` at the repo root)
Next: dogleg_forward.py, with `ki` set to the value chosen here.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from thzim.dogleg import (DoglegGeom, solve_achromat_quads,
                          solve_entrance_twiss)
from thzim.utils import apply_style, output_dir, save

# --------------------------- user settings ---------------------------
REPO = Path(__file__).resolve().parents[2]
FIGS = output_dir(REPO, "OP1", "figures")

GEOM = DoglegGeom(theta_deg=40.0, rho=0.542, delta_x=2.0, d_in=0.2, d_out=0.1)
EKIN_MEV = 15.4             # OP1. The optics is energy-independent; only the
                            # gradients and the R56 velocity term are not.
KI_SCAN = np.linspace(-70.0, -6.0, 33)      # inner-pair strengths [1/m^2]
EMIT_RATIO = 1.0            # eps_y/eps_x; 1.0 = distribution-free scoring
KI_NOMINAL = -38.0          # design of record, drawn for comparison; None = skip
# ---------------------------------------------------------------------


def main():
    energy_gev = (EKIN_MEV + 0.51099895) * 1e-3
    print(f"OP1 dogleg ki scan: theta={GEOM.theta_deg:.2f} deg, "
          f"rho={GEOM.rho:.4f} m, Delta_x={GEOM.delta_x:.2f} m, "
          f"E_kin={EKIN_MEV:.1f} MeV")
    print(f"geometry: L_bend={GEOM.L_bend:.4f}  L_gap={GEOM.L_gap:.4f}  "
          f"d3={GEOM.d3:.4f}  length={GEOM.length:.4f} m  "
          f"(geometry-only R56 = {GEOM.R56_z*1e3:+.2f} mm)\n")

    rows = []
    for ki in KI_SCAN:
        try:
            dl = solve_achromat_quads(GEOM, float(ki), energy_gev=energy_gev)
            ent = solve_entrance_twiss(dl)
        except Exception as exc:                    # infeasible ki
            print(f"  [warn] ki = {ki:+.2f} skipped: {exc}")
            continue
        rows.append(dict(ki=float(ki), ko=dl.ko, r56=dl.R56_z,
                         dx=dl.Dx_exit, dxp=dl.Dxp_exit,
                         beta_x=ent.beta_x, alpha_x=ent.alpha_x,
                         beta_y=ent.beta_y, alpha_y=ent.alpha_y,
                         peak_bx=ent.peak_beta_x, peak_by=ent.peak_beta_y,
                         score=ent.peak_score(EMIT_RATIO)))
    if not rows:
        raise SystemExit("no feasible ki in the scan -- widen KI_SCAN")

    scan = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    best = int(np.argmin(scan["score"]))
    ki_best = scan["ki"][best]

    print(f"{'ki':>8s} {'ko':>9s} {'R56/mm':>9s} {'peak_bx/m':>10s} "
          f"{'peak_by/m':>10s} {'score/m':>9s} {'beta_x/m':>9s} "
          f"{'alpha_x':>9s} {'beta_y/m':>9s} {'alpha_y':>9s}")
    for i, r in enumerate(rows):
        mark = " <-- best" if i == best else ""
        print(f"{r['ki']:+8.2f} {r['ko']:9.4f} {r['r56']*1e3:+9.4f} "
              f"{r['peak_bx']:10.4f} {r['peak_by']:10.4f} {r['score']:9.4f} "
              f"{r['beta_x']:9.4f} {r['alpha_x']:+9.4f} {r['beta_y']:9.4f} "
              f"{r['alpha_y']:+9.4f}{mark}")

    print(f"\nachromat check over the scan: max |Dx_exit| = "
          f"{np.abs(scan['dx']).max()*1e3:.2e} mm, "
          f"max |Dxp_exit| = {np.abs(scan['dxp']).max():.2e}  "
          f"(match tolerance, not physics)")
    print(f"R56: {scan['r56'].min()*1e3:.4f} .. {scan['r56'].max()*1e3:.4f} mm "
          f"(spread {np.ptp(scan['r56'])*1e3:.2e} mm -- ki does not move it)")
    print(f"ko:  {scan['ko'].min():.3f} .. {scan['ko'].max():.3f} 1/m^2 "
          f"(retunes to keep the achromat closed)")
    print(f"\nemit_ratio = {EMIT_RATIO:g}  ->  best ki = {ki_best:+.2f} 1/m^2, "
          f"score = {scan['score'][best]:.4f} m")
    print(f"  entrance Twiss there: beta_x={scan['beta_x'][best]:.4f} m, "
          f"alpha_x={scan['alpha_x'][best]:+.4f}, "
          f"beta_y={scan['beta_y'][best]:.4f} m, "
          f"alpha_y={scan['alpha_y'][best]:+.4f}")
    print(f"  -> set KI = {ki_best:+.2f} in dogleg_forward.py")

    # ------------------------------ plot ------------------------------
    apply_style()
    fig, (ax_s, ax_o) = plt.subplots(figsize=(5.5, 5.5), nrows=2, sharex=True,
                                     layout="constrained")

    ax_s.plot(scan["ki"], scan["peak_bx"], lw=1.4, label=r"peak $\beta_x$")
    ax_s.plot(scan["ki"], scan["peak_by"], lw=1.4, label=r"peak $\beta_y$")
    ax_s.plot(scan["ki"], scan["score"], color="k", lw=1.8, ls="--",
              label=rf"score ($\epsilon_y/\epsilon_x$ = {EMIT_RATIO:g})")
    ax_s.plot(ki_best, scan["score"][best], "o", color="white", markersize=8,
              markeredgecolor="black", markeredgewidth=1.2, zorder=5,
              label=rf"best $k_i$ = {ki_best:+.1f} m$^{{-2}}$")
    if KI_NOMINAL is not None:
        ax_s.axvline(KI_NOMINAL, color="0.4", lw=1.0, ls=":",
                     label=rf"nominal $k_i$ = {KI_NOMINAL:+.1f} m$^{{-2}}$")
    ax_s.set_ylabel(r"peak $\beta$ [$m$]")   # x is shared with (b) below
    ax_s.legend(loc="upper center", fontsize=7, framealpha=0.85)
    ax_s.set_title("(a) what the ki choice costs", fontsize=10)

    ax_o.plot(scan["ki"], scan["ko"], color="C4", lw=1.6)
    ax_o.set_xlabel(r"$k_i$ [$m^{-2}$]")
    ax_o.set_ylabel(r"$k_o$ [$m^{-2}$]", color="C4")
    ax_o.tick_params(axis="y", labelcolor="C4")
    ax_r = ax_o.twinx()
    ax_r.plot(scan["ki"], scan["r56"] * 1e3, color="C1", lw=1.6, ls="--")
    ax_r.set_ylabel(r"$R_{56}$ [$mm$]", color="C1")
    ax_r.tick_params(axis="y", labelcolor="C1")
    # a window wide enough to be a fair test of "flat"; zooming to the actual
    # spread would magnify solver noise into a fake wiggle
    half = max(1.0, 3.0 * np.ptp(scan["r56"]) * 1e3)
    ax_r.set_ylim(scan["r56"].mean() * 1e3 - half,
                  scan["r56"].mean() * 1e3 + half)
    ax_o.set_title(r"(b) $k_o$ retunes with $k_i$; $R_{56}$ does not",
                   fontsize=10)

    plt.show()
    save(fig, FIGS, 'fig_OP1_dogleg_ki_scan')


if __name__ == "__main__":
    main()
