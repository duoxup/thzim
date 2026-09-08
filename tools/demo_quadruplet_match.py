#!/usr/bin/env python3
r"""Quadruplet match demo: four quads onto a target Twiss, linear then under SC.

    launch   d_in  Q1  d  Q2  d  Q3  d  Q4  d_out   target

The OTHER way this package matches a beam (the first is `thzim.two_triplet`).
Four knobs against the four exit Twiss numbers is a square problem, so there is
no scan and no choice for a person to make: `thzim.quadruplet` solves it and
this script shows what the solution does to a real beam. One script, one
figure, an SC switch.

Two layers, run in order:

  1. `solve_linear_match` -- exact transfer-matrix solve from many starts,
     keeping the min-peak-beta solution. Energy-free and instant.
  2. `sc_rematch` (SC = True only) -- damped Newton on the four k1 against the
     SC-TRACKED exit Twiss, Jacobian from the noise-free linear optics, seeded
     from layer 1. This is where the beam enters.

Then the result is tracked end to end and drawn, four panels against s, x blue
and y red:

  (a) normalised emittance -- what the section costs the beam
  (b) beta, tracked (solid) against the linear optics at the SAME k1 (thin
      dashed). With SC off the two coincide, which is the check that the
      tracking and the transfer matrix agree; with SC on the gap between them
      is what space charge does at these knobs. Target marked by black ticks;
      the title carries Bmag (1.0 = matched; the value is the effective
      emittance growth a mismatch costs once it filaments).
  (c) rms size
  (d) the lattice, drawn by ocelot

The case is the OP2 final focus: the dogleg-exit Twiss (10.29, -15.4, 10.44,
-6.3) onto the undulator round waist beta* = 0.53 m, the same numbers the
original study's demos used -- but on a REAL beam rather than a synthetic
Gaussian: the OP2 P0 distribution (1 nC, 40 MeV) re-conditioned to the launch
Twiss by `partdist.match_twiss_xy`, which is symplectic per plane and so keeps
the emittance and the longitudinal profile exactly. Note the current is P0's,
not the 200 A the compressed beam would carry, so the space-charge shift here is
milder than in the design of record. The launch Twiss is then READ BACK from
the distribution rather than taken from the constant, because that is what a
repro script has to do with a beam it did not construct.

`SC = False` runs layer 1 only and tracks without space charge, sampled just as
densely (ocelot subdivides at `unit_step` only where a physics process is
active; `thzim.quadruplet` attaches a no-op one when there is none). It takes
seconds and is the quick look at whether the geometry admits a solution at all
-- `solve_linear_match` raises if it does not -- before paying for the tracked
version. `SC = True` costs a few minutes: every Newton step is an SC track.

Run:  python demo_quadruplet_match.py    (needs `pip install -e .` at the repo
      root plus `partdist`, and data/OP2_50k.dist)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import ocelot as oc

from partdist import read_astra_distribution
from partdist.pd3d.manipulator import match_twiss_xy
from thzim.quadruplet import (QuadrupletGeom, bmag, build_lattice,
                              sc_rematch, solve_linear_match, track_match)
from thzim.triplet import beam_energy_gev
from thzim.utils import apply_style, output_dir, save

# --------------------------- user settings ---------------------------
REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "data" / "OP2_50k.dist"
FIGS = output_dir(REPO, "tools", "figures")

LAUNCH = (10.29, -15.4, 10.44, -6.3)   # dogleg-exit Twiss (bx, ax, by, ay) the
                                       # beam is re-conditioned to
TARGET = (1.53, 0.0, 0.83, 0.0)        # OP2 P3: round waist at the undulator
GEOM = QuadrupletGeom()                # lq 0.10, d_in 0.30, d_inter 0.30,
                                       # d_out 0.40 -- the M3 station, 2.0 m

SC = True                 # False: linear solve + SC-off track (seconds)
                          # True:  + sc_rematch and an SC track (minutes)
SC_MESH = (31, 31, 31)
UNIT_STEP = 0.05          # navigator step [m]; also the twiss sampling density
N_ITER = 8                # Newton steps allowed in sc_rematch
TOL = 5e-4                # on the squared relative residual

C_X, C_Y = "tab:blue", "tab:red"          # x is blue, y is red, everywhere
# ---------------------------------------------------------------------


def sigma_of(tws):
    """s, sigma_x, sigma_y [m] from an ocelot twiss list."""
    s = np.array([t.s for t in tws])
    sx = np.sqrt(np.array([t.emit_x * t.beta_x for t in tws]))
    sy = np.sqrt(np.array([t.emit_y * t.beta_y for t in tws]))
    return s, sx, sy


def beta_linear(match, nsl=20):
    """s, beta_x, beta_y of the LINEAR optics at the match's k1, sliced."""
    lat = build_lattice(match.geom, match.k1s, sliced=True, nsl=nsl)
    t0 = oc.Twiss()
    t0.E = match.energy_gev
    t0.beta_x, t0.alpha_x, t0.beta_y, t0.alpha_y = match.launch
    tws = oc.twiss(lat, t0)
    return (np.array([t.s for t in tws]), np.array([t.beta_x for t in tws]),
            np.array([t.beta_y for t in tws]))


def figure_e2e(match, tws, sc):
    """The matched section, tracked end to end."""
    from ocelot.gui.accelerator import plot_elems

    geom = match.geom
    s, sx, sy = sigma_of(tws)
    enx = np.array([t.emit_xn for t in tws]) * 1e6
    eny = np.array([t.emit_yn for t in tws]) * 1e6
    bx = np.array([t.beta_x for t in tws])
    by = np.array([t.beta_y for t in tws])
    sl, blx, bly = beta_linear(match)
    t1 = tws[-1]
    bmx = bmag(t1.beta_x, t1.alpha_x, match.target[0], match.target[1])
    bmy = bmag(t1.beta_y, t1.alpha_y, match.target[2], match.target[3])

    fig, (ax_e, ax_b, ax_s, ax_l) = plt.subplots(
        figsize=(5.6, 7.4), nrows=4, sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 3, 3, 1]})

    for ax in (ax_e, ax_b, ax_s):
        for s0, s1, _ in geom.element_spans():
            ax.axvspan(s0, s1, color="#8ecae6", alpha=0.45, lw=0, zorder=0)

    # linestyle pinned throughout: colour carries the plane, not the dash
    ax_e.plot(s, enx, color=C_X, lw=1.5, ls="-", label=r"$\epsilon_{n,x}$")
    ax_e.plot(s, eny, color=C_Y, lw=1.5, ls="-", label=r"$\epsilon_{n,y}$")
    ax_e.set_ylabel(r"$\epsilon_n$ [$\mu m$]")
    ax_e.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax_e.set_title(f"(a) normalised emittance "
                   f"(space charge {'ON' if sc else 'OFF'})", fontsize=10)

    ax_b.plot(sl, blx, color=C_X, lw=0.8, ls="--", alpha=0.7)
    ax_b.plot(sl, bly, color=C_Y, lw=0.8, ls="--", alpha=0.7,
              label="linear optics")
    ax_b.plot(s, bx, color=C_X, lw=1.5, ls="-", label=r"$\beta_x$")
    ax_b.plot(s, by, color=C_Y, lw=1.5, ls="-", label=r"$\beta_y$")
    ax_b.plot([s[-1]] * 2, [match.target[0], match.target[2]], "_", color="k",
              ms=12, mew=2.0, zorder=7)
    ax_b.set_ylabel(r"$\beta$ [$m$]")
    ax_b.set_ylim(bottom=0.0)
    ax_b.legend(loc="best", fontsize=8, framealpha=0.85)
    ax_b.set_title(rf"(b) $\beta$, tracked vs linear (black ticks = target; "
                   rf"$B_{{mag}}$ = {bmx:.3f} / {bmy:.3f})", fontsize=10)

    ax_s.plot(s, sx * 1e3, color=C_X, lw=1.5, ls="-", label=r"$\sigma_x$")
    ax_s.plot(s, sy * 1e3, color=C_Y, lw=1.5, ls="-", label=r"$\sigma_y$")
    ax_s.set_ylabel(r"$\sigma$ [$mm$]")
    ax_s.set_ylim(bottom=0.0)
    ax_s.legend(loc="best", fontsize=8, framealpha=0.85)
    ax_s.set_title("(c) rms size", fontsize=10)

    plot_elems(fig, ax_l, match.lattice, legend=False, font_size=8)
    # the scienceplots "ieee" style dashes patch edges, which turns the element
    # blocks into hatching; and the numeric y axis means nothing here
    for patch in ax_l.patches:
        patch.set_linestyle("-")
    ax_l.set_yticks([])
    for side in ("left", "right", "top"):
        ax_l.spines[side].set_visible(False)
    ax_l.set_xlim(-0.02 * geom.length, 1.02 * geom.length)
    ax_l.set_xlabel(r"$s$ [$m$]   (launch $\to$ Q1 Q2 Q3 Q4 $\to$ target)")
    return fig


def main():
    dist = read_astra_distribution(str(DIST))
    beam = match_twiss_xy(dist, alpha_x=LAUNCH[1], beta_x=LAUNCH[0],
                          alpha_y=LAUNCH[3], beta_y=LAUNCH[2])
    tw = beam.twiss                                 # read back, not assumed
    launch = (tw["beta_x"], tw["alpha_x"], tw["beta_y"], tw["alpha_y"])
    energy_gev = beam_energy_gev(beam)

    print(GEOM.summary())
    print(f"  beam: {DIST.name} re-conditioned to LAUNCH, n={len(beam)}, "
          f"Q={abs(beam.get_data('Q').sum())*1e9:.3f} nC, "
          f"E={energy_gev*1e3:.2f} MeV, I_peak={beam.I_peak:.0f} A")
    print(f"  launch (read from the beam): bx={launch[0]:.3f} "
          f"ax={launch[1]:+.3f}  by={launch[2]:.3f} ay={launch[3]:+.3f}")
    print(f"  target: bx={TARGET[0]:.3f} ax={TARGET[1]:+.3f}  "
          f"by={TARGET[2]:.3f} ay={TARGET[3]:+.3f}"
          f"   (space charge {'ON' if SC else 'OFF'})\n")

    lin = solve_linear_match(GEOM, launch, TARGET, energy_gev=energy_gev)
    print(lin.summary())

    if SC:
        print("\nSC re-match (every line below is one SC track):")
        match = sc_rematch(lin, beam, n_iter=N_ITER, tol=TOL,
                           unit_step=UNIT_STEP, nmesh=SC_MESH)
        print()
        print(match.summary())
        pre = match.seed_exit_twiss
        print(f"\n  at the LINEAR k1 under SC the exit was bx={pre[0]:.3f} "
              f"ax={pre[1]:+.3f}  by={pre[2]:.3f} ay={pre[3]:+.3f}   "
              f"Bmag={bmag(pre[0], pre[1], *TARGET[:2]):.3f}/"
              f"{bmag(pre[2], pre[3], *TARGET[2:]):.3f}")
        print(f"  {'':10s}"
              + "".join(f"{q:>10s}" for q in ("Q1", "Q2", "Q3", "Q4"))
              + "   [1/m^2]")
        print("  linear    " + "".join(f"{k:+10.3f}" for k in lin.k1s))
        print("  SC        " + "".join(f"{k:+10.3f}" for k in match.k1s))
        print("  moved by  " + "".join(f"{b - a:+10.3f}"
                                       for a, b in zip(lin.k1s, match.k1s)))
    else:
        match = lin

    print("\ntracking end to end ...")
    exit_tw, tws = track_match(match.lattice, beam, sc=SC,
                               unit_step=UNIT_STEP, nmesh=SC_MESH)
    t0, t1 = tws[0], tws[-1]
    s, sx, sy = sigma_of(tws)
    bmx = bmag(exit_tw[0], exit_tw[1], TARGET[0], TARGET[1])
    bmy = bmag(exit_tw[2], exit_tw[3], TARGET[2], TARGET[3])
    print(f"  delivered  bx={exit_tw[0]:.3f} ax={exit_tw[1]:+.3f}  "
          f"by={exit_tw[2]:.3f} ay={exit_tw[3]:+.3f}   "
          f"Bmag={bmx:.3f}/{bmy:.3f}")
    print(f"  emittance  eps_n,x {t0.emit_xn*1e6:.4f} -> "
          f"{t1.emit_xn*1e6:.4f} um   eps_n,y {t0.emit_yn*1e6:.4f} -> "
          f"{t1.emit_yn*1e6:.4f} um")
    print(f"  beam size  peak ({sx.max()*1e3:.3f}, {sy.max()*1e3:.3f}) mm; "
          f"exit ({sx[-1]*1e3:.3f}, {sy[-1]*1e3:.3f}) mm")

    apply_style()
    save(figure_e2e(match, tws, SC), FIGS, 'fig_demo_quadruplet_match')
    plt.show()


if __name__ == "__main__":
    main()
