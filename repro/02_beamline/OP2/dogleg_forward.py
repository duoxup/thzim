#!/usr/bin/env python3
r"""OP2 dogleg — step 2 of 2: the solved lattice at one chosen ki.

Takes the ki picked in dogleg_ki_scan.py, re-runs the two solves, and shows what
the resulting dogleg actually does: Twiss and dispersion from entrance to exit.

  API 1  solve_achromat_quads(geom, ki) -> ko, and the exit Dx / Dx'
  API 2  solve_entrance_twiss(dl)       -> the entrance Twiss to be delivered by
                                           the upstream match
  then   twiss_along(dl, entrance)      -> the forward propagation drawn here

That last step derives nothing new. The peak betas were already found inside
API 2 (its min-peak search ran the full Twiss for every trial centre beta), and
the exit dispersion already came out of API 1 -- dispersion is driven by the
dipoles from Dx = Dx' = 0 at the entrance, so it never depended on the entrance
beta/alpha at all. This script is the visualisation and the self-check: the
peak of the curve in (a) must reproduce the solver's peak_beta, and (b) must
come back to zero. Both are asserted below.

Two panels, with the bends and both quad families shaded behind:
  (a) beta_x, beta_y from entrance to exit
  (b) Dx -- built up by B1, unwound by B2, closed at the exit

The entrance Twiss printed here is the hand-off to the upstream match
(`thzim.two_triplet`, run by `two_triplet_scan.py` in this directory), which is
where a real distribution finally enters the chain.

OP2 runs the same dogleg hardware as OP1 at 39.4 MeV instead of 15.4, and every
Twiss quantity drawn here is bit-identical to OP1's at the same ki: `k1` is a
geometric strength and the bend is given by angle, so nothing in the transfer
matrix carries energy. What differs is `ki` -- OP1's -38 was set by hand, OP2's
-22 is the ki_scan winner -- which is why the two figures look nothing alike
even though neither depends on the beam energy. See docs/dogleg_design.md.

Run:  python dogleg_forward.py    (needs `pip install -e .` at the repo root)
Prev: dogleg_ki_scan.py, which is where KI below comes from.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from thzim.dogleg import (DoglegGeom, element_spans, solve_achromat_quads,
                          solve_entrance_twiss, twiss_along)
from thzim.utils import apply_style, output_dir, save

# --------------------------- user settings ---------------------------
REPO = Path(__file__).resolve().parents[3]
FIGS = output_dir(REPO, "OP2", "figures")

GEOM = DoglegGeom(theta_deg=40.0, rho=0.542, delta_x=2.0, d_in=0.0, d_out=0.3)
EKIN_MEV = 39.4             # OP2
KI = -22.0                  # design of record, and the min-peak choice at
                            # emit_ratio = 1 (see dogleg_ki_scan.py)
NSL = 30                    # slices per element for the forward Twiss
BAND_COLORS = {"bend": "#c8c8c8", "QO": "#8ecae6", "QI": "#ffb703"}
# ---------------------------------------------------------------------


def main():
    energy_gev = (EKIN_MEV + 0.51099895) * 1e-3

    dl = solve_achromat_quads(GEOM, KI, energy_gev=energy_gev)
    ent = solve_entrance_twiss(dl)
    s, bx, by, dx, dxp = twiss_along(dl, ent, nsl=NSL)

    print(dl.summary())
    print("-" * 72)
    print(ent.summary())

    # self-check: the forward pass must reproduce what the solvers reported
    print("\nforward check (nothing new should appear here)")
    for plane, curve, solved in (("x", bx, ent.peak_beta_x),
                                 ("y", by, ent.peak_beta_y)):
        peak, at = curve.max(), s[int(np.argmax(curve))]
        print(f"  peak beta_{plane}: forward {peak:8.4f} m at s = {at:.3f} m  "
              f"vs solver {solved:8.4f} m   (diff {peak-solved:+.2e})")
    print(f"  exit Dx = {dx[-1]*1e3:+.3e} mm, Dx' = {dxp[-1]:+.3e}  "
          f"-> {'ACHROMATIC' if dl.achromatic else 'NOT closed'}")
    print(f"  peak |Dx| inside the dogleg = {np.abs(dx).max()*1e3:.1f} mm "
          f"at s = {s[int(np.argmax(np.abs(dx)))]:.3f} m")
    print(f"\nhand this entrance Twiss to the upstream match: "
          f"{tuple(round(float(v), 5) for v in ent.as_tuple())}")

    # ------------------------------ plot ------------------------------
    apply_style()
    fig, (ax_b, ax_d) = plt.subplots(figsize=(5.5, 5.5), nrows=2, sharex=True,
                                     layout="constrained")

    for ax in (ax_b, ax_d):
        for s0, s1, kind in element_spans(GEOM):
            ax.axvspan(s0, s1, color=BAND_COLORS[kind], alpha=0.35, lw=0)
    ax_d.set_xlim(0.0, GEOM.length)
    ax_d.set_xlabel(r"$s$ [$m$]")          # shared, so only the bottom axes

    ax_b.plot(s, bx, lw=1.6, label=r"$\beta_x$")
    ax_b.plot(s, by, lw=1.6, label=r"$\beta_y$")
    ax_b.set_ylabel(r"$\beta$ [$m$]")
    ax_b.legend(loc="upper center", fontsize=8, framealpha=0.85)
    ax_b.set_title(rf"(a) $k_i$ = {dl.ki:+.1f} m$^{{-2}}$, "
                   rf"$k_o$ = {dl.ko:+.2f} m$^{{-2}}$", fontsize=10)

    ax_d.plot(s, dx * 1e3, color="C2", lw=1.6)
    ax_d.axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax_d.set_ylabel(r"$D_x$ [$mm$]")
    ax_d.set_title(rf"(b) achromat closure: exit $D_x$ = "
                   rf"{dl.Dx_exit*1e3:+.1e} mm", fontsize=10)

    plt.show()
    save(fig, FIGS, 'fig_OP2_dogleg_forward')


if __name__ == "__main__":
    main()
