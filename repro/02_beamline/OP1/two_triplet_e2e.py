#!/usr/bin/env python3
r"""OP1 upstream match — step 2 of 2: run the chosen working point end to end.

    P0   T1(Q1 Q2 Q3)   the chicane crossed as a drift   T2(Q1 Q2 Q3)   dogleg

Takes the GRADIENTS picked off `two_triplet_scan.py` -- all six, not just the
two free ones -- builds the full line in physical order and tracks the P0 beam
through it. Nothing is solved here.

Why the gradients and not the (g1_fwd, g1_bwd) that were scanned: **a g1 does
not determine the lattice.** The round solve at a fixed g1 has several branches,
and which one it reaches depends on the seed. The scan finds the right one by
continuity from its neighbours; a cold solve at the same g1 has nothing to be
continuous with. At g1_bwd = +0.3542 that difference is T2 = (-0.579, +0.582,
+0.354) T/m from the scan against (+0.357, +5.648, +0.354) cold -- a 106 1/m^2
quadrupole, a free-screen residual of 180 % against 0.00 %, and sigma_y blowing
up to 35 mm past T2. Copying the gradients removes the ambiguity. Four panels against s, x blue and y red:

  (a) normalised emittance -- what the section costs the beam
  (b) beta, with the dogleg entrance target marked; the title carries Bmag,
      which is the number that says whether the match worked (1.0 is matched,
      and the value is the effective emittance growth a mismatch causes once it
      filaments)
  (c) rms size, round across the shaded middle by construction. Watch this one:
      a hard waist in the middle is where a 1 nC beam at 15.9 MeV loses its
      emittance, and it is why the choice is made on figures rather than by a
      solver.
  (d) the lattice, drawn by ocelot

The scan script does not choose, so this one has no search in it. It also
re-prints the free-screen residual at this working point -- the check that the
two legs really describe one beam, since two screen readings alone cannot pin
it. Several per cent there and Bmag will be off by a similar amount; narrow the
g1 windows in the scan script rather than fighting it here.

The linear crossing and its SC-corrected counterpart are both recorded in the
settings block. The SC solution is active by default. With `SC = True`, the
tracked P1 distribution is written to `outputs/OP1/beams/p1.ast` for the
compressor stage.

`SC = False` tracks the same lattice with space charge off, sampled just as
densely. Use it to separate what the OPTICS does from what the CHARGE does --
the linear track is also the honest reference for the emittance panel, since the
P0 beam is chirped (sigma_p/p = 1.3 %, full spread 5.3 %) and picks up chromatic
projected-emittance structure in strong quads that has nothing to do with space
charge.

Run:  python two_triplet_e2e.py     (needs `pip install -e .` at the repo root
      plus `partdist`)
Prev: two_triplet_scan.py, which is where T1_G / T2_G come from.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from partdist import (from_ocelot_particle_array, read_astra_distribution,
                      write_astra_distribution)
from thzim.dogleg import DoglegGeom, solve_achromat_quads, solve_entrance_twiss
from thzim.two_triplet import (TwoTripletGeom, build_lattice, match_with,
                               track_e2e)
from thzim.utils import apply_style, output_dir, save

# --------------------------- user settings ---------------------------
REPO = Path(__file__).resolve().parents[3]
DIST = REPO / "data" / "OP1_50k.dist"
FIGS = output_dir(REPO, "OP1", "figures")
P1_DIST = output_dir(REPO, "OP1", "beams") / "p1.ast"

EKIN_MEV = 15.4                 # OP1
DOGLEG = DoglegGeom(theta_deg=40.0, rho=0.542, delta_x=2.0, d_in=0.2, d_out=0.1)
KI = -38.0                      # OP1 design of record; sets the target Twiss

GEOM = TwoTripletGeom(t1_lq=(0.10, 0.10, 0.10), t1_drifts=(0.30, 0.30, 0.30),
                      t2_lq=(0.10, 0.10, 0.10), t2_drifts=(0.30, 0.30, 0.40),
                      gap=3.40, screens=(0.34, 1.70, 3.06))

# The working point, copied from two_triplet_scan.py. GRADIENTS, not just g1:
# the round solve at a fixed g1 has several branches and a cold call
# lands on whichever the seed reaches, which is not the one the scan
# crossed on. See thzim.two_triplet.match_with.
# Linear crossing (`SC_SCAN = False`, `SC_KNOBS = False`):
# T1_G = (-0.2427, +0.4705, -0.2405)
# T2_G = (-0.5994, +0.6747, +0.2992)

# SC-corrected crossing (`SC_SCAN = True`, `SC_KNOBS = False`), default:
T1_G = (-0.2647, +0.5131, -0.2649)
T2_G = (-0.6401, +0.7404, +0.2494)
     
SCREEN_PAIR = (0, -1)     # must match the scan, so the residual means the same

SC = True                 # space charge in the e2e track AND in the screen
                          # readings printed below
SC_MESH = (31, 31, 31)
UNIT_STEP = 0.05          # navigator step [m]. It also sets how densely the
                          # twiss output is sampled, in BOTH modes -- see
                          # thzim.two_triplet._Sampler

C_X, C_Y = "blue", "red"          # x is blue, y is red, everywhere
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
    ax_b.set_title(rf"(b) $\beta$ (black ticks = dogleg entrance; "
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
    ax_l.set_xlabel(r"$s$ [$m$]   (P0 $\to$ T1 $\to$ screens $\to$ T2 "
                    r"$\to$ dogleg)")
    return fig


def main():
    energy_gev = (EKIN_MEV + 0.51099895) * 1e-3
    with contextlib.redirect_stdout(io.StringIO()):
        dl = solve_achromat_quads(DOGLEG, KI, energy_gev=energy_gev)
        target = solve_entrance_twiss(dl).as_tuple()
    dist = read_astra_distribution(str(DIST))
    print(GEOM.summary())
    print(f"  beam: {DIST.name}, n={len(dist)}, "
          f"Q={abs(dist.get_data('Q').sum())*1e9:.3f} nC, "
          f"E={dist.gamma0 * 0.51099895:.2f} MeV")
    print(f"  target (dogleg entrance, ki={KI:+.1f}): bx={target[0]:.4f} "
          f"ax={target[1]:+.4f}  by={target[2]:.4f} ay={target[3]:+.4f}")
    print("  working point: T1 = ("
          + ", ".join(f"{g:+.4f}" for g in T1_G) + "), T2 = ("
          + ", ".join(f"{g:+.4f}" for g in T2_G) + ") T/m"
          f"   (space charge {'ON' if SC else 'OFF'})\n")

    match = match_with(dist, GEOM, target, T1_G, T2_G, sc=SC,
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
    tracked = track_e2e(match, dist, sc=SC, unit_step=UNIT_STEP, mesh=SC_MESH,
                        return_particles=SC)
    if SC:
        tws, pa_out = tracked
        P1_DIST.parent.mkdir(parents=True, exist_ok=True)
        write_astra_distribution(P1_DIST, from_ocelot_particle_array(pa_out))
        print(f"  wrote P1 distribution: {P1_DIST}")
    else:
        tws = tracked
    print()
    print(match.summary())
    t0, t1 = tws[0], tws[-1]
    s, sx, sy = sigma_of(tws)
    i = int(np.argmin(sx + sy))
    print(f"  emittance  eps_n,x {t0.emit_xn*1e6:.4f} -> {t1.emit_xn*1e6:.4f} um"
          f"   eps_n,y {t0.emit_yn*1e6:.4f} -> {t1.emit_yn*1e6:.4f} um")
    print(f"  beam size  peak ({sx.max()*1e3:.3f}, {sy.max()*1e3:.3f}) mm; "
          f"tightest ({sx[i]*1e3:.3f}, {sy[i]*1e3:.3f}) mm at s = {s[i]:.2f} m")

    apply_style()
    save(figure_e2e(match, tws, SC), FIGS, 'fig_OP1_two_triplet_e2e')
    plt.show()


if __name__ == "__main__":
    main()
