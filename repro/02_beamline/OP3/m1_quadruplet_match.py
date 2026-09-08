#!/usr/bin/env python3
r"""OP3 M1 quadruplet match: P0 -> chicane entrance (P1).

The superradiant branch matches the P0 beam onto the chicane entrance Twiss
with the four-quad M1 station. The target combines a hard solve with the
adopted horizontal design choice:

  x   alpha_x = 0, beta_x = 30 m. The value comes from the soft trade-off in
      `chicane_beta_x_scan.py` and is the formal nominal choice, even though
      nothing in linear theory pins it.
  y   the analytic edge-focusing matched solution of the chicane,
      `thzim.chicane.matched_y` with lead = 0.2 m, evaluated at OP3's chicane
      geometry.  Injecting that Twiss keeps beta_y flat across the chicane.

So the script computes the y target itself from the same `ChicaneGeom` the
`chicane_beta_x_scan.py` run uses, then solves the square four-quad problem

    launch (measured P0 Twiss)  ->  M1  ->  target Twiss

in two layers:

  1. `solve_linear_match` -- exact transfer-matrix solve from many starts,
     keeping the min-peak-beta solution.  Energy-free and instant.
  2. `sc_rematch` (SC = True only) -- damped Newton on the four k1 against the
     SC-TRACKED exit Twiss, Jacobian from the noise-free linear optics, seeded
     from layer 1.  This is where the beam enters.

Then the matched line is tracked end to end and drawn.  SC is on by default;
set `SC = False` for a quick linear check. With `SC = True`, the tracked P1
distribution is written to `outputs/OP3/beams/p1.ast`.

Run:  python m1_quadruplet_match.py    (needs `pip install -e .` at the repo
      root plus `partdist`)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from partdist import (from_ocelot_particle_array, read_astra_distribution,
                      write_astra_distribution)
from thzim.chicane import ChicaneGeom, matched_y
from thzim.quadruplet import (QuadrupletGeom, bmag, build_lattice,
                              sc_rematch, solve_linear_match, track_match)
from thzim.triplet import beam_energy_gev
from thzim.utils import apply_style, output_dir, save

# --------------------------- user settings ---------------------------
REPO = Path(__file__).resolve().parents[3]
DIST = REPO / "data" / "OP3_50k.dist"
FIGS = output_dir(REPO, "OP3", "figures")
P1_DIST = output_dir(REPO, "OP3", "beams") / "p1.ast"

# The chicane entrance plane is 0.2 m upstream of its first bend.
CHICANE = ChicaneGeom(theta_deg=16.69, l_bz=0.10, l_dz=0.75, lead=0.20)

# x target: nominal selected from the beta_x tracking scan.
# y target: computed below from `matched_y`; do not set by hand.
BETA_X_TARGET = 30.0
ALPHA_X_TARGET = 0.0

GEOM = QuadrupletGeom()                # M1: lq 0.10, d_in 0.30, d_inter 0.30,
                                       # d_out 0.40 -> total length 2.0 m

SC = True                 # False: linear solve + SC-off track (seconds)
                          # True:  + sc_rematch and an SC track (minutes)
SC_MESH = (63, 63, 63)
UNIT_STEP = 0.02          # navigator step [m]; also the twiss sampling density
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
    import ocelot as oc
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
    for patch in ax_l.patches:
        patch.set_linestyle("-")
    ax_l.set_yticks([])
    for side in ("left", "right", "top"):
        ax_l.spines[side].set_visible(False)
    ax_l.set_xlim(-0.02 * geom.length, 1.02 * geom.length)
    ax_l.set_xlabel(r"$s$ [$m$]   (P0 $\to$ Q1 Q2 Q3 Q4 $\to$ P1)")
    return fig


def main():
    dist = read_astra_distribution(str(DIST))
    tw = dist.twiss
    launch = (tw["beta_x"], tw["alpha_x"], tw["beta_y"], tw["alpha_y"])
    energy_gev = beam_energy_gev(dist)

    my = matched_y(CHICANE, energy_gev)
    if my is None:
        raise SystemExit(f"y channel unstable at theta = {CHICANE.theta_deg} deg")
    beta_y_target, alpha_y_target, mu_y = my
    target = (BETA_X_TARGET, ALPHA_X_TARGET, beta_y_target, alpha_y_target)

    print(CHICANE.summary())
    print(f"  matched y: beta*={beta_y_target:.4f} m  alpha*={alpha_y_target:+.4f}  "
          f"mu={mu_y:.2f} deg   (lead = {CHICANE.lead})")
    print(GEOM.summary())
    print(f"  beam: {DIST.name}, n={len(dist)}, "
          f"Q={abs(dist.get_data('Q').sum())*1e9:.3f} nC, "
          f"E={energy_gev*1e3:.2f} MeV, I_peak={dist.I_peak:.0f} A")
    print(f"  launch (P0, read from the beam): bx={launch[0]:.3f} "
          f"ax={launch[1]:+.3f}  by={launch[2]:.3f} ay={launch[3]:+.3f}")
    print(f"  target: bx={target[0]:.3f} ax={target[1]:+.3f}  "
          f"by={target[2]:.4f} ay={target[3]:+.4f}"
          f"   (space charge {'ON' if SC else 'OFF'})\n")

    lin = solve_linear_match(GEOM, launch, target, energy_gev=energy_gev)
    print(lin.summary())

    if SC:
        print("\nSC re-match (every line below is one SC track):")
        match = sc_rematch(lin, dist, n_iter=N_ITER, tol=TOL,
                           unit_step=UNIT_STEP, nmesh=SC_MESH)
        print()
        print(match.summary())
        pre = match.seed_exit_twiss
        print(f"\n  at the LINEAR k1 under SC the exit was bx={pre[0]:.3f} "
              f"ax={pre[1]:+.3f}  by={pre[2]:.3f} ay={pre[3]:+.3f}   "
              f"Bmag={bmag(pre[0], pre[1], *target[:2]):.3f}/"
              f"{bmag(pre[2], pre[3], *target[2:]):.3f}")
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
    tracked = track_match(match.lattice, dist, sc=SC, unit_step=UNIT_STEP,
                          nmesh=SC_MESH, return_particles=SC)
    if SC:
        exit_tw, tws, pa_out = tracked
        P1_DIST.parent.mkdir(parents=True, exist_ok=True)
        write_astra_distribution(P1_DIST, from_ocelot_particle_array(pa_out))
        print(f"  wrote P1 distribution: {P1_DIST}")
    else:
        exit_tw, tws = tracked
    t0, t1 = tws[0], tws[-1]
    s, sx, sy = sigma_of(tws)
    bmx = bmag(exit_tw[0], exit_tw[1], target[0], target[1])
    bmy = bmag(exit_tw[2], exit_tw[3], target[2], target[3])
    print(f"  delivered  bx={exit_tw[0]:.3f} ax={exit_tw[1]:+.3f}  "
          f"by={exit_tw[2]:.4f} ay={exit_tw[3]:+.3f}   "
          f"Bmag={bmx:.3f}/{bmy:.3f}")
    print(f"  emittance  eps_n,x {t0.emit_xn*1e6:.4f} -> "
          f"{t1.emit_xn*1e6:.4f} um   eps_n,y {t0.emit_yn*1e6:.4f} -> "
          f"{t1.emit_yn*1e6:.4f} um")
    print(f"  beam size  peak ({sx.max()*1e3:.3f}, {sy.max()*1e3:.3f}) mm; "
          f"exit ({sx[-1]*1e3:.3f}, {sy[-1]*1e3:.3f}) mm")

    apply_style()
    save(figure_e2e(match, tws, SC), FIGS, 'fig_OP3_m1_quadruplet_match')
    plt.show()


if __name__ == "__main__":
    main()
