#!/usr/bin/env python3
r"""OP4 M3 quadruplet match: chicane exit (P2) -> undulator entrance (P3).

    P2  [T1.Q1 off]  [T1.Q2 off]  T1.Q3   switch region 0.92 m   T2.Q1 Q2 Q3   P3

The superradiant branch's final focus. The hardware after the chicane is two
triplets around the 0.92 m switch region (M2 and M3-SR in the layout), six
quads for four exit Twiss numbers. The design of record runs it as a FOUR-quad
match instead: T1's first two quads are switched off, and T1.Q3 plus the T2
triplet form a quadruplet with gaps (0.92, 0.30, 0.30) m, solved on the
transfer matrix by `thzim.quadruplet` exactly as M1 is.

Why not the two-triplet crossing route (`thzim.two_triplet`) that the SASE
branch uses upstream: that route rests on each leg carrying a ROUND beam
across the gap, and the P2 beam is not one. The chicane leaves it with
eps_n,x about 4x eps_n,y (space charge and CSR in the bends -- OP4 is the
collective-dominated point) and a residual dispersion of ~100 mm, so no
setting of T1 makes sigma_x = sigma_y in the switch region without a hard
waist, and the scan finds no crossing. A joint
six-knob solve could get round this by weighting the planes with the
emittance ratio, i.e. by asking for equal BETA rather than equal size, but
that is not a physical roundness at all. Four quads against four numbers is
a square problem and needs no such assumption. Switching off the two nearest
P2 rather than any other pair costs nothing in the linear solve (the peak
betas are within a metre of the alternative) and keeps the powered quads
downstream, where the beam is already sized for the target.

The P3 target is the FEL-side choice recorded in the design of record and
carried here as a constant: (beta_x, alpha_x, beta_y, alpha_y) =
(9.8, 0, 0.006, 0), a vertical waist of 6 mm at the undulator entrance. There
is no derivation script for it. A 6 mm waist 0.4 m past the last quad is
chromatic even at OP4's 0.4 % energy spread; the Bmag printed is on the
projected Twiss, as in the design of record, and reports that residual.

Matching is done on the PROJECTED P2 Twiss, as in the design of record. The
projected horizontal Twiss includes the residual dispersion; the dispersion-
corrected values are printed next to it so the difference can be seen.

Two layers, then a track:

  1. `solve_linear_match` -- exact transfer-matrix solve from many starts,
     keeping the min-peak-beta solution. Energy-free and instant.
  2. `sc_rematch` (SC = True only) -- damped Newton on the four k1 against the
     SC-TRACKED exit Twiss, Jacobian from the noise-free linear optics, seeded
     from layer 1.

Then the matched line is tracked end to end and drawn (x blue, y red): (a)
normalised emittance, (b) beta tracked against linear with the target marked
and Bmag in the title, (c) rms size, (d) the lattice. The two switched-off
slots are hatched. With `SC = True` the tracked P3 distribution is written to
`outputs/OP4/beams/p3.ast` as the handover to the FEL simulation (delivered
separately), and its longitudinal summary (peak current, bunch length, energy
spread) is printed for it.

Input:  `outputs/OP4/beams/p2.ast` (from `chicane_track.py`)
Output: `outputs/OP4/beams/p3.ast`

Run:  python m3_quadruplet_match.py   (needs `pip install -e .` at the repo
      root plus `partdist`; a few minutes with SC on)
Prev: chicane_track.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from partdist import (from_ocelot_particle_array, read_astra_distribution,
                      write_astra_distribution)
from thzim.compressor import beam_report
from thzim.quadruplet import (QuadrupletGeom, bmag, build_lattice,
                              sc_rematch, solve_linear_match, track_match)
from thzim.triplet import as_ocelot, beam_energy_gev
from thzim.utils import apply_style, output_dir, save

# --------------------------- user settings ---------------------------
REPO = Path(__file__).resolve().parents[2]
P2_DIST = output_dir(REPO, "OP4", "beams") / "p2.ast"
P3_DIST = output_dir(REPO, "OP4", "beams") / "p3.ast"
FIGS = output_dir(REPO, "OP4", "figures")

# P3 target (beta_x, alpha_x, beta_y, alpha_y): the FEL-side choice of record.
TARGET = (9.8, 0.0, 0.006, 0.0)

# T1.Q3 + T2 as a quadruplet. T1's first two slots are switched off and fold
# into d_in: 0.10 + 0.10 + 0.30 + 0.10 + 0.30 = 0.90 m to T1.Q3.
GEOM = QuadrupletGeom(lq=0.10, d_in=0.90, d_inter=(0.92, 0.30, 0.30),
                      d_out=0.40)
DEAD_SLOTS = ((0.10, 0.20), (0.50, 0.60))      # T1.Q1, T1.Q2 (s from P2)

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
        for s0, s1 in DEAD_SLOTS:
            ax.axvspan(s0, s1, facecolor="none", edgecolor="0.6", hatch="///",
                       lw=0, zorder=0)

    ax_e.plot(s, enx, color=C_X, lw=1.5, ls="-", label=r"$\epsilon_{n,x}$")
    ax_e.plot(s, eny, color=C_Y, lw=1.5, ls="-", label=r"$\epsilon_{n,y}$")
    ax_e.set_ylabel(r"$\epsilon_n$ [$\mu m$]")
    ax_e.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax_e.set_title(f"(a) normalised emittance, projected "
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
    ax_l.set_xlabel(r"$s$ [$m$]   (P2 $\to$ T1.Q3 $\to$ switch region "
                    r"$\to$ T2 $\to$ P3; hatched = switched off)")
    return fig


def print_beam(label, r):
    """One beam's longitudinal and transverse summary, for the FEL handover."""
    print(f"  {label:<4s} sig_z={r['sig_z']*1e3:.4f} mm  I_pk={r['I_peak']:.0f} A  "
          f"sig_dp={r['sig_dp']*100:.3f} %  chirp={-r['chirp']:+.2f} /m  "
          f"enx={r['enx']*1e6:.3f} ({r['enx_c']*1e6:.3f} corr) um  "
          f"eny={r['eny']*1e6:.3f} um  eta={r['eta']*1e3:+.2f} mm")


def main():
    dist = read_astra_distribution(str(P2_DIST))
    tw = dist.twiss
    launch = (tw["beta_x"], tw["alpha_x"], tw["beta_y"], tw["alpha_y"])
    energy_gev = beam_energy_gev(dist)
    r_in = beam_report(as_ocelot(dist))

    print(GEOM.summary())
    print("  T1.Q1 and T1.Q2 switched off (slots at s = "
          + ", ".join(f"{a:.2f}-{b:.2f}" for a, b in DEAD_SLOTS) + " m)")
    print(f"  beam: {P2_DIST.relative_to(REPO)}, n={len(dist)}, "
          f"Q={abs(dist.get_data('Q').sum())*1e9:.3f} nC, "
          f"E={energy_gev*1e3:.2f} MeV, I_peak={r_in['I_peak']:.0f} A")
    print(f"  launch (P2, projected): bx={launch[0]:.3f} ax={launch[1]:+.3f}  "
          f"by={launch[2]:.3f} ay={launch[3]:+.3f}")
    print(f"  (dispersion-corrected:  bx={r_in['beta_x']:.3f} "
          f"ax={r_in['alpha_x']:+.3f}; residual eta {r_in['eta']*1e3:+.2f} mm, "
          f"sigma_delta {r_in['sig_dp']*100:.2f} %)")
    print(f"  target (P3): bx={TARGET[0]:.3f} ax={TARGET[1]:+.3f}  "
          f"by={TARGET[2]:.4f} ay={TARGET[3]:+.4f}"
          f"   (space charge {'ON' if SC else 'OFF'})\n")

    lin = solve_linear_match(GEOM, launch, TARGET, energy_gev=energy_gev)
    print(lin.summary())

    if SC:
        print("\nSC re-match (every line below is one SC track):")
        match = sc_rematch(lin, dist, n_iter=N_ITER, tol=TOL,
                           unit_step=UNIT_STEP, nmesh=SC_MESH)
        print()
        print(match.summary())
        pre = match.seed_exit_twiss
        print(f"\n  at the LINEAR k1 under SC the exit was bx={pre[0]:.3f} "
              f"ax={pre[1]:+.3f}  by={pre[2]:.4f} ay={pre[3]:+.3f}   "
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
    tracked = track_match(match.lattice, dist, sc=SC, unit_step=UNIT_STEP,
                          nmesh=SC_MESH, return_particles=True)
    exit_tw, tws, pa_out = tracked
    if SC:
        P3_DIST.parent.mkdir(parents=True, exist_ok=True)
        write_astra_distribution(P3_DIST, from_ocelot_particle_array(pa_out))
        print(f"  wrote P3 distribution: {P3_DIST.relative_to(REPO)}")
    t0, t1 = tws[0], tws[-1]
    s, sx, sy = sigma_of(tws)
    bmx = bmag(exit_tw[0], exit_tw[1], TARGET[0], TARGET[1])
    bmy = bmag(exit_tw[2], exit_tw[3], TARGET[2], TARGET[3])
    print(f"  delivered  bx={exit_tw[0]:.3f} ax={exit_tw[1]:+.3f}  "
          f"by={exit_tw[2]:.4f} ay={exit_tw[3]:+.3f}   Bmag={bmx:.3f}/{bmy:.3f}")
    print(f"  emittance  eps_n,x {t0.emit_xn*1e6:.4f} -> "
          f"{t1.emit_xn*1e6:.4f} um   eps_n,y {t0.emit_yn*1e6:.4f} -> "
          f"{t1.emit_yn*1e6:.4f} um   (projected)")
    print(f"  beam size  peak ({sx.max()*1e3:.3f}, {sy.max()*1e3:.3f}) mm; "
          f"exit ({sx[-1]*1e3:.3f}, {sy[-1]*1e3:.3f}) mm")
    print("\nP3 handover summary (chirp against z, head positive):")
    print_beam("P2", r_in)
    print_beam("P3", beam_report(pa_out))

    apply_style()
    save(figure_e2e(match, tws, SC), FIGS, 'fig_OP4_m3_quadruplet_match')
    plt.show()


if __name__ == "__main__":
    main()
