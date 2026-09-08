#!/usr/bin/env python3
r"""Two-triplet match, step 2 of 2: run the chosen working point end to end.

Takes the (g1_fwd, g1_bwd) pair picked off `demo_two_triplet_scan.py`, re-solves
each leg's round knobs there, builds the full line in physical order and tracks
the injection beam through it:

    inj   T1(Q1 Q2 Q3)   GAP (3 screens)   T2(Q1 Q2 Q3)   target

Four panels against s, x blue and y red:

  (a) normalised emittance -- what the section costs the beam
  (b) beta, with the target marked; the title carries Bmag, which is the number
      that says whether the match worked (1.0 is matched, and the value is the
      effective emittance growth a mismatch causes once it filaments)
  (c) rms size, round across the shaded middle by construction
  (d) the lattice, drawn by ocelot

The scan script does not choose for you, so this one has no search in it: it
does exactly what the two numbers below say. It also re-prints the free-screen
residual at this working point -- the check that the two legs really describe
one beam, since two screen readings alone cannot pin it. If that residual is
several per cent, expect Bmag to be off by a similar amount, and narrow the g1
windows in the scan script rather than fighting it here.

`SC = False` tracks the same lattice with space charge off, sampled just as
densely (ocelot subdivides at `unit_step` only where a physics process is
active, so `thzim.two_triplet.track_e2e` attaches a no-op one when there is no
space charge; without it the curves come out element-granular whatever
`unit_step` says). Use it to separate what the OPTICS does from what the CHARGE
does -- the linear track is also the
honest reference for the emittance panel, since a chirped beam through strong
quads picks up chromatic projected-emittance structure that has nothing to do
with space charge.

Run:  python demo_two_triplet_e2e.py    (needs `pip install -e .` at the repo
      root plus `partdist`)
Prev: demo_two_triplet_scan.py, which is where T1_G / T2_G come from.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from partdist import read_astra_distribution
from thzim.two_triplet import (TwoTripletGeom, build_lattice, match_with,
                               track_e2e)
from thzim.utils import apply_style, output_dir, save

# --------------------------- user settings ---------------------------
REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "data" / "OP1_50k.dist"
FIGS = output_dir(REPO, "tools", "figures")

GEOM = TwoTripletGeom(t1_lq=(0.15, 0.15, 0.15), t1_drifts=(0.50, 0.15, 0.15),
                      t2_lq=(0.15, 0.15, 0.15), t2_drifts=(0.15, 0.15, 0.20),
                      gap=3.0, screens=(0.30, 1.50, 2.70))
TARGET = (16.672, 32.527, 2.180, 2.339)   # OP1 dogleg entrance (bx, ax, by, ay)

# The working point, copied from demo_two_triplet_scan.py. GRADIENTS, not just g1:
# the round solve at a fixed g1 has several branches and a cold call
# lands on whichever the seed reaches, which is not the one the scan
# crossed on. See thzim.two_triplet.match_with.
T1_G = (-0.2347, +0.4550, -0.2318)   # T1 (Q1, Q2, Q3) [T/m]
T2_G = (-0.5792, +0.5817, +0.3542)   # T2 (Q1, Q2, Q3) [T/m]

T1_G = (-0.2440, +0.4780, -0.2477)
T2_G = (-0.1116, -0.3840, +0.8123)
     
SCREEN_PAIR = (0, -1)     # must match the scan, so the residual means the same

SC = False                # space charge in the e2e track AND in the screen
                          # readings printed below
SC_MESH = (31, 31, 31)
UNIT_STEP = 0.05          # navigator step [m]. It also sets how densely the
                          # twiss output is sampled, in BOTH modes -- see
                          # thzim.two_triplet._Sampler

C_X, C_Y = "tab:blue", "tab:red"          # x is blue, y is red, everywhere
# ---------------------------------------------------------------------


def sigma_of(tws):
    """s, sigma_x, sigma_y [m] from an ocelot twiss list."""
    s = np.array([t.s for t in tws])
    sx = np.sqrt(np.array([t.emit_x * t.beta_x for t in tws]))
    sy = np.sqrt(np.array([t.emit_y * t.beta_y for t in tws]))
    return s, sx, sy


def figure_e2e(match, tws, sc):
    """The matched line, tracked end to end."""
    from ocelot.gui.accelerator import plot_elems

    geom = match.geom
    lat = build_lattice(geom, match.t1_g, match.t2_g, match.energy_gev)
    s, sx, sy = sigma_of(tws)
    enx = np.array([t.emit_xn for t in tws]) * 1e6
    eny = np.array([t.emit_yn for t in tws]) * 1e6
    bx = np.array([t.beta_x for t in tws])
    by = np.array([t.beta_y for t in tws])

    fig, (ax_e, ax_b, ax_s, ax_l) = plt.subplots(
        figsize=(5.6, 7.4), nrows=4, sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 3, 3, 1]})

    for ax in (ax_e, ax_b, ax_s):
        ax.axvspan(geom.s_gap_start, geom.s_gap_end, color="#c8c8c8",
                   alpha=0.30, lw=0, zorder=0)
        for sc_s in geom.screen_s:
            ax.axvline(sc_s, color="0.4", ls=":", lw=0.8)

    # linestyle pinned throughout: colour carries the plane, not the dash
    ax_e.plot(s, enx, color=C_X, lw=1.5, ls="-", label=r"$\epsilon_{n,x}$")
    ax_e.plot(s, eny, color=C_Y, lw=1.5, ls="-", label=r"$\epsilon_{n,y}$")
    ax_e.set_ylabel(r"$\epsilon_n$ [$\mu m$]")
    ax_e.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax_e.set_title(f"(a) normalised emittance "
                   f"(space charge {'ON' if sc else 'OFF'})", fontsize=10)

    ax_b.plot(s, bx, color=C_X, lw=1.5, ls="-", label=r"$\beta_x$")
    ax_b.plot(s, by, color=C_Y, lw=1.5, ls="-", label=r"$\beta_y$")
    ax_b.plot([s[-1]] * 2, [match.target[0], match.target[2]], "_", color="k",
              ms=12, mew=2.0, zorder=7)
    ax_b.set_ylabel(r"$\beta$ [$m$]")
    ax_b.set_ylim(bottom=0.0)
    bmx, bmy = match.bmag()
    ax_b.legend(loc="best", fontsize=8, framealpha=0.85)
    ax_b.set_title(rf"(b) $\beta$ (black ticks = target; "
                   rf"$B_{{mag}}$ = {bmx:.3f} / {bmy:.3f})", fontsize=10)

    ax_s.plot(s, sx * 1e3, color=C_X, lw=1.5, ls="-", label=r"$\sigma_x$")
    ax_s.plot(s, sy * 1e3, color=C_Y, lw=1.5, ls="-", label=r"$\sigma_y$")
    ax_s.set_ylabel(r"$\sigma$ [$mm$]")
    ax_s.set_ylim(bottom=0.0)
    ax_s.legend(loc="best", fontsize=8, framealpha=0.85)
    ax_s.set_title("(c) rms size (round across the shaded middle)", fontsize=10)

    plot_elems(fig, ax_l, lat, legend=False, font_size=8)
    # the scienceplots "ieee" style dashes patch edges, which turns the element
    # blocks into hatching; and the numeric y axis means nothing here
    for patch in ax_l.patches:
        patch.set_linestyle("-")
    ax_l.set_yticks([])
    for side in ("left", "right", "top"):
        ax_l.spines[side].set_visible(False)
    ax_l.set_xlim(-0.02 * geom.length, 1.02 * geom.length)
    ax_l.set_xlabel(r"$s$ [$m$]   (injection $\to$ T1 $\to$ screens $\to$ T2 "
                    r"$\to$ target)")
    return fig


def main():
    dist = read_astra_distribution(str(DIST))
    print(GEOM.summary())
    print(f"  beam: {DIST.name}, n={len(dist)}, "
          f"Q={abs(dist.get_data('Q').sum())*1e9:.3f} nC, "
          f"E={dist.gamma0 * 0.51099895:.2f} MeV")
    print("  working point: T1 = ("
          + ", ".join(f"{g:+.4f}" for g in T1_G) + "), T2 = ("
          + ", ".join(f"{g:+.4f}" for g in T2_G) + ") T/m"
          f"   (space charge {'ON' if SC else 'OFF'})\n")

    match = match_with(dist, GEOM, TARGET, T1_G, T2_G, sc=SC,
                       screen_pair=SCREEN_PAIR, unit_step=UNIT_STEP,
                       mesh=SC_MESH)
    print("screen readings at this working point [mm]:")
    print(f"  {'screen':>7s} {'s/m':>6s} {'forward':>9s} {'backward':>9s} "
          f"{'diff':>8s}")
    n_scr = len(GEOM.screens)
    pair = (SCREEN_PAIR[0] % n_scr, SCREEN_PAIR[1] % n_scr)
    for c in range(n_scr):
        sf, sb = match.fwd_sigma[c], match.bwd_sigma[c]
        role = "matched" if c in pair else "FREE (the check)"
        print(f"  {c:>7d} {GEOM.screen_s[c]:6.2f} {sf*1e3:9.4f} {sb*1e3:9.4f} "
              f"{abs(sf-sb)/(0.5*(sf+sb))*100:7.2f}%   {role}")

    print("\ntracking end to end ...")
    tws = track_e2e(match, dist, sc=SC, unit_step=UNIT_STEP, mesh=SC_MESH)
    print()
    print(match.summary())
    t0, t1 = tws[0], tws[-1]
    print(f"  emittance  eps_n,x {t0.emit_xn*1e6:.4f} -> {t1.emit_xn*1e6:.4f} um"
          f"   eps_n,y {t0.emit_yn*1e6:.4f} -> {t1.emit_yn*1e6:.4f} um")

    apply_style()
    save(figure_e2e(match, tws, SC), FIGS, 'fig_demo_two_triplet_e2e')
    plt.show()


if __name__ == "__main__":
    main()
