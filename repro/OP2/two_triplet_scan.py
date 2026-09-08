#!/usr/bin/env python3
r"""OP2 upstream match — step 1 of 2: scan both legs into the dogleg.

The SASE branch delivers the P0 beam to the dogleg entrance through two quad
stations with the (switched-off) chicane footprint between them:

    P0   T1(Q1 Q2 Q3)   the chicane crossed as a drift   T2(Q1 Q2 Q3)   dogleg

Six gradients, four of them spent on making each leg's beam ROUND, leaving two:
T1's first quad and T2's last. Because both legs arrive round, agreeing on the
beam SIZE at two screens in the middle means agreeing on the full covariance in
both planes -- two knobs for four Twiss constraints. This script scans those two
and shows what each choice does. It does NOT choose: read the figures, pick a
row, and put its gradients into `two_triplet_e2e.py`.

The target is not a free parameter -- it is the dogleg entrance Twiss, computed
here from `thzim.dogleg` exactly as `dogleg_forward.py` does. The magnets are
the same as OP1, at 39.4 MeV instead of 15.4 and at the scan-optimum ki = -22
instead of the hand-set -38. OP2 takes the dogleg entrance marker at the first
bend (`d_in = 0`), so its target is defined at that plane.

## What is different from OP1

The P0 beam. At OP1 it arrives at beta = 49 m, sigma = 2.3 mm, diverging; at
OP2 it arrives at beta = 0.34 m, sigma = 0.10 mm, CONVERGING -- it goes through
its own waist (0.094 mm at s = 0.14 m, still 0.16 m short of Q1) before this
section can touch it. Nothing here changes that waist; the pinch to watch for
is the one in the MIDDLE. And the gradients are an order of magnitude larger:
2.5x the rigidity, and a beam that has to be caught right after a waist.

With `d_in = 0`, the useful crossing lies on the NEGATIVE T2 branch near
`g1_bwd = -5.6 T/m`; a scan limited to the old positive branch cannot see it.
The analytic scan has several crossings, so the independent middle screen is
essential. On the selected weak-T1 branch the SC scan leaves one consistent
candidate, with a 0.42 % middle-screen residual.

## What to look for when choosing

Not every intersection is a match, and among the real ones not every one is
good.

* **Is it real?** Two screen readings do not pin a beam: in a drift sigma^2(s)
  is a parabola, so at fixed emittance two free parameters remain and a
  different (waist size, waist position) can pass through the same two points.
  The screens NOT used to form the crossing are the check, and the table's
  `free scr` column reports it.
* **Is it good?** A real crossing can still drive a hard waist in the middle,
  which at 1 nC is where the emittance would be lost even at 40 MeV. The
  backward leg here is steep: across the local window its first-screen size
  falls from about 0.96 to 0.08 mm. The selected crossing stays away from that
  end of the branch; figure 1 makes this check visible before the e2e run.

A curve can have gaps where no round solution was found. `leg_scan` tries three
seeds per point (the previous point's answer, the given seed, and its mirror);
what is left after that is genuinely infeasible and is dropped to `nan`.

Space charge: both leg scans track with SC. The round knobs (g2, g3) remain a
LINEAR solve: explicitly enabling `SC_KNOBS` did not materially move this
crossing, while the selected candidate is already round to 2.0 % under SC.
Every scan point still reports the measured SC roundness as a diagnostic.

`SC_SCAN = False` runs the whole scan analytically in seconds -- use it to check
the windows first; its candidates are NOT the answer, and it reports no
roundness column because there is no track to read it from.

Run:  python two_triplet_scan.py    (needs `pip install -e .` at the repo root
      plus `partdist`; ~8 min with SC_SCAN = True, ~10 s without)
Next: two_triplet_e2e.py, with the gradients chosen here.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from partdist import read_astra_distribution
from thzim.dogleg import DoglegGeom, solve_achromat_quads, solve_entrance_twiss
from thzim.maps import drift_map, propagate, quad_map
from thzim.triplet import injection_sigma
from thzim.two_triplet import (TwoTripletGeom, backward_beam, find_crossings,
                               leg_scan, match_at)
from thzim.utils import apply_style, output_dir, save

# --------------------------- user settings ---------------------------
REPO = Path(__file__).resolve().parents[2]
DIST = REPO / "data" / "OP2_50k.dist"
FIGS = output_dir(REPO, "OP2", "figures")

EKIN_MEV = 39.4                 # OP2
DOGLEG = DoglegGeom(theta_deg=40.0, rho=0.542, delta_x=2.0, d_in=0.0, d_out=0.3)
KI = -22.0                      # OP2 design of record (the ki-scan optimum);
                                # sets the target Twiss

# The section in physical order, with the same hardware as OP1. T1 is a
# triplet in a four-slot station, so its dead fourth slot folds into the gap:
# 0.30 + 0.10 + 3.00 = 3.40 m. Screens sit at 10 %, 50 % and 90 % of the gap.
GEOM = TwoTripletGeom(t1_lq=(0.10, 0.10, 0.10), t1_drifts=(0.30, 0.30, 0.30),
                      t2_lq=(0.10, 0.10, 0.10), t2_drifts=(0.30, 0.30, 0.40),
                      gap=3.40,
                      screens=(0.34, 1.70, 3.06))

G1_FWD = np.linspace(-3.50, -2.00, 11)    # T1 first quad [T/m]
G1_BWD = np.linspace(-5.75, -5.45, 11)    # T2 last quad  [T/m]
SCREEN_PAIR = (0, -1)                     # the rest are the independent check

SC_SCAN = True                            # False = analytic dry run
SC_KNOBS = False                          # verified sufficient for this branch
SC_MESH = (31, 31, 31)
SC_UNIT_STEP = 0.05
# ---------------------------------------------------------------------


def sigma_of(tws):
    """s, sigma_x, sigma_y [m] from an ocelot twiss list."""
    s = np.array([t.s for t in tws])
    sx = np.sqrt(np.array([t.emit_x * t.beta_x for t in tws]))
    sy = np.sqrt(np.array([t.emit_y * t.beta_y for t in tws]))
    return s, sx, sy


def envelope_linear(leg, k, sig0_x, sig0_y, nsl=40):
    """s, sigma_x, sigma_y [m] through a leg + its exit drift, analytic 2x2.

    The counterpart of `sigma_of` for the no-tracking mode; `thzim.maps` gives
    the few lines it takes, so it does not need to live in the package.
    """
    lq, d = leg.lq, leg.drifts
    elems = [(None, d[0]), (k[0], lq[0]), (None, d[1]), (k[1], lq[1]),
             (None, d[2]), (k[2], lq[2]), (None, d[3])]
    Sx, Sy = sig0_x.copy(), sig0_y.copy()
    s, sx, sy, pos = [0.0], [np.sqrt(Sx[0, 0])], [np.sqrt(Sy[0, 0])], 0.0
    for ki, length in elems:
        dl = length / nsl
        Mx = drift_map(dl) if ki is None else quad_map(+ki, dl)
        My = drift_map(dl) if ki is None else quad_map(-ki, dl)
        for _ in range(nsl):
            Sx, Sy = propagate(Mx, Sx), propagate(My, Sy)
            pos += dl
            s.append(pos)
            sx.append(np.sqrt(Sx[0, 0]))
            sy.append(np.sqrt(Sy[0, 0]))
    return np.array(s), np.array(sx), np.array(sy)


def _leg_panel(ax, rows, leg, offsets, norm, cmap, beam, xlabel):
    """sigma(s) per scanned g1 through one leg, quads and screens marked."""
    s0x, s0y = injection_sigma(beam)
    for s0, s1, _ in leg.element_spans():
        ax.axvspan(s0, s1, color="#8ecae6", alpha=0.45, lw=0, zorder=0)
    ax.axvspan(leg.s_q3_exit, leg.length, color="#c8c8c8", alpha=0.30, lw=0,
               zorder=0)
    for row in rows:
        if not np.isfinite(row["sigma"][0]):
            continue                      # no round solution at this g1
        if row.get("tws") is not None:
            s, sx, sy = sigma_of(row["tws"])
        else:
            s, sx, sy = envelope_linear(leg, row["k"], s0x, s0y)
        c = cmap(norm(row["g1"]))
        # linestyle pinned explicitly: colour carries the plane, dash the curve kind
        ax.plot(s, sx * 1e3, color=c, lw=1.2, ls="-")
        ax.plot(s, sy * 1e3, color=c, lw=1.0, ls="--")
    for off in offsets:
        ax.axvline(leg.s_q3_exit + off, color="0.4", ls=":", lw=0.8)
    ax.set_xlim(0.0, leg.length)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel)


def figure_legs(fwd_rows, bwd_rows, dist, bwd, sc):
    """Figure 1: the two leg scans."""
    fig, (ax_f, ax_b) = plt.subplots(figsize=(6.6, 3.2), ncols=2, sharey=True,
                                     layout="constrained")
    cmap = plt.get_cmap("viridis")
    nf = mpl.colors.Normalize(min(G1_FWD), max(G1_FWD))
    nb = mpl.colors.Normalize(min(G1_BWD), max(G1_BWD))
    # each panel runs in its OWN leg's direction, so the backward one reads
    # right-to-left against the machine; the labels say so
    _leg_panel(ax_f, fwd_rows, GEOM.t1_leg, GEOM.screens, nf, cmap, dist,
               r"$s$ from P0 [$m$]")
    _leg_panel(ax_b, bwd_rows, GEOM.t2_leg, GEOM.bwd_screen_offsets, nb, cmap,
               bwd, r"$s$ from the dogleg, upstream [$m$]")
    how = "SC-tracked" if sc else "analytic"
    ax_f.set_ylabel(r"$\sigma$ [$mm$]")
    ax_f.set_title(f"(a) forward leg T1, from P0 ({how})", fontsize=10)
    ax_b.set_title(f"(b) backward leg T2, from the dogleg ({how})", fontsize=10)
    ax_f.legend(handles=[Line2D([], [], color="0.3", lw=1.2, ls="-",
                                label=r"$\sigma_x$"),
                         Line2D([], [], color="0.3", lw=1.0, ls="--",
                                label=r"$\sigma_y$")],
                loc="upper left", fontsize=8, framealpha=0.85)
    fig.colorbar(mpl.cm.ScalarMappable(norm=nf, cmap=cmap), ax=ax_f,
                 label=r"$g_1$ [$T/m$]")
    fig.colorbar(mpl.cm.ScalarMappable(norm=nb, cmap=cmap), ax=ax_b,
                 label=r"$g_1$ [$T/m$]")
    return fig


def figure_crossing(fwd_rows, bwd_rows, cands, pair):
    """Figure 2: the parametric screen-size map and the candidate crossings."""
    i, j = pair
    f = np.array([r["sigma"] for r in fwd_rows])[:, [i, j]] * 1e3
    b = np.array([r["sigma"] for r in bwd_rows])[:, [i, j]] * 1e3
    fig, ax = plt.subplots(figsize=(4.6, 4.3), layout="constrained")
    ax.plot(f[:, 0], f[:, 1], color="tab:red", marker="o", ms=3.0, lw=1.2,
            ls="-", label="forward (T1)")
    ax.plot(b[:, 0], b[:, 1], color="tab:blue", marker="o", ms=3.0, lw=1.2,
            ls="-", label="backward (T2)")
    for n, c in enumerate(cands):
        x, y = c["sigma"][0] * 1e3, c["sigma"][1] * 1e3
        ax.plot(x, y, "o", ms=10, mfc="none", mec="k", mew=1.4, zorder=6)
        res = (f"{c['residual']*100:.1f}%" if np.isfinite(c["residual"])
               else "n/a")
        ax.annotate(f"#{n}  ({res})", xy=(x, y), xytext=(10, 8),
                    textcoords="offset points", fontsize=7.5, zorder=6)
    ax.set_xlabel(rf"$\sigma$ @ screen {i} "
                  rf"($s$ = {GEOM.screen_s[i]:.2f} m) [$mm$]")
    ax.set_ylabel(rf"$\sigma$ @ screen {j} "
                  rf"($s$ = {GEOM.screen_s[j]:.2f} m) [$mm$]")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.set_title("candidate working points\n"
                 "(% = how far the free screen disagrees; small is a real match)",
                 fontsize=10)
    return fig


def main():
    energy_gev = (EKIN_MEV + 0.51099895) * 1e-3
    with contextlib.redirect_stdout(io.StringIO()):      # the solver is chatty
        dl = solve_achromat_quads(DOGLEG, KI, energy_gev=energy_gev)
        target = solve_entrance_twiss(dl).as_tuple()

    print(f"target = the OP2 dogleg entrance Twiss at ki = {KI:+.1f} m^-2:")
    print(f"  bx={target[0]:.4f} ax={target[1]:+.4f}  "
          f"by={target[2]:.4f} ay={target[3]:+.4f}")
    print()

    dist = read_astra_distribution(str(DIST))
    n_scr = len(GEOM.screens)
    pair = (SCREEN_PAIR[0] % n_scr, SCREEN_PAIR[1] % n_scr)
    free = [c for c in range(n_scr) if c not in pair]
    print(GEOM.summary())
    print(f"  beam: {DIST.name}, n={len(dist)}, "
          f"Q={abs(dist.get_data('Q').sum())*1e9:.3f} nC, "
          f"E={dist.gamma0 * 0.51099895:.2f} MeV")
    print(f"  crossing formed from screens {pair}, checked against {free}")
    print(f"  space charge in the scans: "
          f"{'ON' if SC_SCAN else 'OFF (dry run -- candidates are NOT the answer)'}"
          f",  round knobs: {'SC' if SC_KNOBS else 'linear'}\n")

    bwd = backward_beam(dist, target)
    kw = dict(sc=SC_SCAN, sc_knobs=SC_KNOBS, unit_step=SC_UNIT_STEP,
              mesh=SC_MESH, keep_tws=SC_SCAN, verbose=True)
    print(f"  scanning forward leg ({len(G1_FWD)} points) ...")
    fwd_rows = leg_scan(dist, GEOM.t1_leg, G1_FWD, tuple(GEOM.screens), **kw)
    print(f"  scanning backward leg ({len(G1_BWD)} points) ...")
    bwd_rows = leg_scan(bwd, GEOM.t2_leg, G1_BWD,
                        tuple(GEOM.bwd_screen_offsets), **kw)

    F = np.array([r["sigma"] for r in fwd_rows])
    B = np.array([r["sigma"] for r in bwd_rows])
    raw = find_crossings(F[:, list(pair)], G1_FWD, B[:, list(pair)], G1_BWD)

    def nearest_seed(rows, g):
        good = [r for r in rows if np.isfinite(r["sigma"][0])]
        if not good:
            return (-2.0, 0.0)
        near = min(good, key=lambda r: abs(r["g1"] - g))
        return (near["g2"], near["g3"])

    cands = []
    for gf, gb, sa, sc_ in raw:
        # re-evaluate both legs AT the candidate rather than interpolating the
        # scan, seeded from the nearest scan point so the round solve stays on
        # the same branch
        m = match_at(dist, GEOM, target, gf, gb, sc=SC_SCAN,
                     sc_knobs=SC_KNOBS, screen_pair=pair,
                     unit_step=SC_UNIT_STEP, mesh=SC_MESH, bwd=bwd,
                     g_seed=nearest_seed(fwd_rows, gf),
                     g_seed_bwd=nearest_seed(bwd_rows, gb))
        cands.append(dict(g1_fwd=gf, g1_bwd=gb, sigma=(sa, sc_),
                          residual=m.screen_residual, match=m))

    print(f"\n{len(cands)} candidate working point(s):\n")
    if cands:
        print(f"{'#':>2s} {'g1_fwd':>8s} {'g1_bwd':>8s} "
              f"{'sig@S%d/mm' % pair[0]:>10s} {'sig@S%d/mm' % pair[1]:>10s} "
              f"{'free scr':>9s} {'SC round':>9s}   "
              f"T1 (Q1,Q2,Q3) / T2 (Q1,Q2,Q3) [T/m]")
        for n, c in enumerate(cands):
            m = c["match"]
            res = (f"{c['residual']*100:8.2f}%" if np.isfinite(c["residual"])
                   else "  UNUSABLE")
            rnd = (f"{m.round_dev*100:8.1f}%" if m.round_dev is not None
                   and np.isfinite(m.round_dev) else "        -")
            print(f"{n:>2d} {c['g1_fwd']:+8.4f} {c['g1_bwd']:+8.4f} "
                  f"{c['sigma'][0]*1e3:10.4f} {c['sigma'][1]*1e3:10.4f} "
                  f"{res} {rnd}   "
                  + "(" + ", ".join(f"{g:+.3f}" for g in m.t1_g) + ") / ("
                  + ", ".join(f"{g:+.3f}" for g in m.t2_g) + ")")
        print("\n'free scr' is how far the two legs disagree at the screen(s) "
              "NOT used to form the\ncrossing. A few per cent is a real match; "
              "tens of per cent is a parabola coincidence.\nOf the real ones, "
              "prefer the candidate whose figure-1 envelopes do NOT pinch to a "
              "tiny\nwaist in the middle -- that is where a 1 nC beam at 40 "
              "MeV loses its emittance.")
        devs = [c["match"].round_dev for c in cands
                if c["match"].round_dev is not None
                and np.isfinite(c["match"].round_dev)]
        if devs and max(devs) >= 0.05:
            print(f"\n!! 'SC round' reaches {max(devs)*100:.0f} % -- the beam is "
                  f"NOT round in the middle under space\n"
                  f"   charge, so two screen readings do not pin it and these "
                  f"crossings do not mean what\n"
                  f"   they should. Set SC_KNOBS = True and re-run before "
                  f"choosing.")
        print("\n-> choose a row and copy its GRADIENTS into two_triplet_e2e.py:\n")
        for n, c in enumerate(cands):
            m = c["match"]
            print(f"   #{n}   T1_G = ("
                  + ", ".join(f"{g:+.4f}" for g in m.t1_g) + ")\n"
                  + "        T2_G = ("
                  + ", ".join(f"{g:+.4f}" for g in m.t2_g) + ")")
        print("\n   All six, not just the two scanned ones: a g1 does not "
              "determine the lattice.\n   The round solve at a fixed g1 has "
              "several branches, and this scan picked its\n   branch by "
              "continuity from the neighbouring points -- a cold solve at the "
              "same g1\n   has nothing to be continuous with and can land "
              "somewhere else entirely.")
    else:
        print("  none -- the curves do not meet in these windows. Widen or move "
              "G1_FWD / G1_BWD\n  (figure 1 shows which g1 are even round; try "
              "SC_SCAN = False first, it takes seconds).")

    # ------------------------------ plots ------------------------------
    apply_style()
    save(figure_legs(fwd_rows, bwd_rows, dist, bwd, SC_SCAN), FIGS,
         'fig_OP2_two_triplet_legs')
    save(figure_crossing(fwd_rows, bwd_rows, cands, pair), FIGS,
         'fig_OP2_two_triplet_crossing')
    plt.show()


if __name__ == "__main__":
    main()
