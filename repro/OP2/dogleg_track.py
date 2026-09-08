#!/usr/bin/env python3
r"""OP2 dogleg: track the matched P1 beam through the compressor to P2.

    P1   d_in  B1(+t)  d1  QO  d2  QI  d3 | C | d3  QI  d2  QO  d1  B2(-t)  d_out   P2

The lattice is the OP2 design of record: the `DoglegGeom` and `ki` below are
the ones `two_triplet_e2e.py` computed its entrance target from, and `ko`
is re-solved from the achromat condition exactly as there (`thzim.dogleg.
solve_achromat_quads`), so the target and the lattice cannot drift apart.
Nothing else is solved: the beam written by `two_triplet_e2e.py` is read back
and tracked with space charge and CSR, and the result is what the final
match (M3) will see.

Two tracks are run on the same beam and drawn together:

  * the LINEAR reference -- SC and CSR off. The compression itself is optics
    (R56 times the chirp), so this is the bunch length the lattice delivers
    on its own, and the residual dispersion the achromat match leaves.
  * the COLLECTIVE track -- SC + CSR on. The difference is what the charge
    costs: the emittance it adds, the dispersion it leaves unclosed, the
    energy it radiates away, and how much of the compression it spoils. At
    1 nC and 39 MeV it is milder than at OP1, but the same magnets are
    2.5x stronger in T/m and the entrance marker sits AT the first bend
    (`d_in = 0`), so there is no lead-in drift to let the beam relax.

Figure 1, five panels against s (x blue, y red; dotted = linear reference):

  (a) rms size: the total sigma_x and its BETATRON part, dispersion removed.
      Between the dipoles eta reaches ~0.2 m and the dispersive term
      dominates, so the total says nothing about the optics; the betatron
      size is what the quads and the space charge act on.
  (b) the statistical dispersion eta(s), closing to its exit residual.
  (c) normalised emittance: eps_n,x DISPERSION-CORRECTED, and eps_n,y. The
      projected eps_n,x is not drawn -- it swings up and back with eta, which
      is bookkeeping, not growth; its exit value is in the table.
  (d) the compression: sigma_z on the left axis, peak current on the right.
  (e) the lattice, drawn by ocelot.

Figure 2 is the longitudinal phase space at P1 and P2 with the current profile
under each, the direct picture of what the compressor did to the chirp.

Input:  `outputs/OP2/beams/p1.ast` (from `two_triplet_e2e.py`, SC on)
Output: `outputs/OP2/beams/p2.ast`, consumed by the final match.

Run:  python dogleg_track.py     (needs `pip install -e .` at the repo root
      plus `partdist`; ~1 min with SC + CSR at 50k particles)
Prev: two_triplet_e2e.py
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from partdist import (from_ocelot_particle_array, read_astra_distribution,
                      write_astra_distribution)
from thzim.compressor import beam_report, track_compressor
from thzim.dogleg import DoglegGeom, element_spans, solve_achromat_quads
from thzim.triplet import as_ocelot, beam_energy_gev
from thzim.utils import apply_style, output_dir, save

# --------------------------- user settings ---------------------------
REPO = Path(__file__).resolve().parents[2]
P1_DIST = output_dir(REPO, "OP2", "beams") / "p1.ast"
P2_DIST = output_dir(REPO, "OP2", "beams") / "p2.ast"
FIGS = output_dir(REPO, "OP2", "figures")

# OP2 design of record; must match two_triplet_e2e.py, whose entrance target
# was computed from this geometry and ki.
DOGLEG = DoglegGeom(theta_deg=40.0, rho=0.542, delta_x=2.0, d_in=0.0, d_out=0.3)
KI = -22.0                      # inner-pair k1 [1/m^2]; ko follows from the achromat

SC = True                 # space charge in the collective track
CSR = True                # coherent synchrotron radiation in the collective track
SC_MESH = (63, 63, 63)
UNIT_STEP = 0.02          # navigator step [m]; also the record spacing
CSR_NBIN = 300

C_X, C_Y, C_Z = "tab:blue", "tab:red", "tab:green"   # x blue, y red, z green
# ---------------------------------------------------------------------


def print_report(label, r, ref=None):
    """One line per beam; `ref` (the entrance) turns sigma_z into a ratio."""
    comp = f"  C={ref['sig_z'] / r['sig_z']:.2f}" if ref else ""
    print(f"  {label:<14s} sig_z={r['sig_z']*1e3:.4f} mm{comp}  "
          f"I_pk={r['I_peak']:.0f} A  sig_dp={r['sig_dp']*100:.3f} %  "
          f"chirp={-r['chirp']:+.2f} /m  dE={r['mean_dp']*r['E_mev']*1e3:+.1f} keV")
    print(f"  {'':14s} enx={r['enx']*1e6:.3f} ({r['enx_c']*1e6:.3f} corr) um  "
          f"eny={r['eny']*1e6:.3f} um  eta={r['eta']*1e3:+.2f} mm  "
          f"bx={r['beta_x']:.2f} ax={r['alpha_x']:+.3f}  "
          f"by={r['beta_y']:.3f} ay={r['alpha_y']:+.3f}")


def figure_evolution(evo, evo0, lattice, physics):
    """The compressor, tracked: sizes, dispersion, emittance, compression."""
    from ocelot.gui.accelerator import plot_elems

    s, s0 = evo["s"], evo0["s"]
    fig, (ax_s, ax_d, ax_e, ax_z, ax_l) = plt.subplots(
        figsize=(5.6, 9.6), nrows=5, sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 3, 3, 3, 1]})

    for ax in (ax_s, ax_d, ax_e, ax_z):
        for a, b, kind in element_spans(DOGLEG):
            ax.axvspan(a, b, color="#c8c8c8" if kind == "bend" else "#8ecae6",
                       alpha=0.40, lw=0, zorder=0)

    # linestyle explicit throughout: colour = plane, dash = which curve
    ax_s.plot(s, evo["sig_x"] * 1e3, color=C_X, lw=1.5, ls="-",
              label=r"$\sigma_x$ total")
    ax_s.plot(s, evo["sig_xb"] * 1e3, color=C_X, lw=1.2, ls="--",
              label=r"$\sigma_x$ betatron")
    ax_s.plot(s, evo["sig_y"] * 1e3, color=C_Y, lw=1.5, ls="-",
              label=r"$\sigma_y$")
    ax_s.plot(s0, evo0["sig_xb"] * 1e3, color=C_X, lw=0.9, ls=":")
    ax_s.plot(s0, evo0["sig_y"] * 1e3, color=C_Y, lw=0.9, ls=":")
    ax_s.set_ylabel(r"$\sigma$ [$mm$]")
    ax_s.set_ylim(bottom=0.0)
    ax_s.legend(loc="upper left", fontsize=7, framealpha=0.85)
    ax_s.set_title("(a) rms size (dotted = linear reference)", fontsize=10)

    ax_d.plot(s, evo["eta"] * 1e3, color=C_X, lw=1.5, ls="-", label="tracked")
    ax_d.plot(s0, evo0["eta"] * 1e3, color=C_X, lw=0.9, ls=":", label="linear")
    ax_d.axhline(0.0, color="0.5", lw=0.6)
    ax_d.set_ylabel(r"$\eta_x$ [$mm$]")
    ax_d.legend(loc="upper left", fontsize=7, framealpha=0.85)
    ax_d.set_title(rf"(b) dispersion; exit residual {evo['eta'][-1]*1e3:+.2f} mm "
                   rf"(linear {evo0['eta'][-1]*1e3:+.2f} mm)", fontsize=10)

    ax_e.plot(s, evo["enx_c"] * 1e6, color=C_X, lw=1.5, ls="-",
              label=r"$\epsilon_{n,x}$ (dispersion-corrected)")
    ax_e.plot(s, evo["eny"] * 1e6, color=C_Y, lw=1.5, ls="-",
              label=r"$\epsilon_{n,y}$")
    ax_e.plot(s0, evo0["enx_c"] * 1e6, color=C_X, lw=0.9, ls=":")
    ax_e.plot(s0, evo0["eny"] * 1e6, color=C_Y, lw=0.9, ls=":")
    ax_e.set_ylabel(r"$\epsilon_n$ [$\mu m$]")
    ax_e.legend(loc="upper left", fontsize=7, framealpha=0.85)
    ax_e.set_title("(c) normalised emittance "
                   f"({physics})",
                   fontsize=10)

    ax_z.plot(s, evo["sig_z"] * 1e3, color=C_Z, lw=1.5, ls="-",
              label=r"$\sigma_z$")
    ax_z.plot(s0, evo0["sig_z"] * 1e3, color=C_Z, lw=0.9, ls=":")
    ax_z.set_ylabel(r"$\sigma_z$ [$mm$]")
    ax_z.set_ylim(bottom=0.0)
    ax_i = ax_z.twinx()
    ax_i.plot(s, evo["I_peak"], color="k", lw=1.2, ls="-", label=r"$I_{peak}$")
    ax_i.plot(s0, evo0["I_peak"], color="k", lw=0.8, ls=":")
    ax_i.set_ylabel(r"$I_{peak}$ [$A$]")
    ax_i.set_ylim(bottom=0.0)
    h1, l1 = ax_z.get_legend_handles_labels()
    h2, l2 = ax_i.get_legend_handles_labels()
    ax_z.legend(h1 + h2, l1 + l2, loc="center left", fontsize=7, framealpha=0.85)
    ax_z.set_title(rf"(d) compression: $\sigma_z$ {evo['sig_z'][0]*1e3:.3f} "
                   rf"$\to$ {evo['sig_z'][-1]*1e3:.3f} mm "
                   rf"(linear {evo0['sig_z'][-1]*1e3:.3f} mm)", fontsize=10)

    plot_elems(fig, ax_l, lattice, legend=False, font_size=8)
    for patch in ax_l.patches:
        patch.set_linestyle("-")
    ax_l.set_yticks([])
    for side in ("left", "right", "top"):
        ax_l.spines[side].set_visible(False)
    ax_l.set_xlim(-0.02 * DOGLEG.length, 1.02 * DOGLEG.length)
    ax_l.set_xlabel(r"$s$ [$m$]   (P1 $\to$ B1 QO QI $|$ QI QO B2 $\to$ P2)")
    return fig


def figure_lps(pa_in, pa_out, nbin=200):
    """Longitudinal phase space at P1 and P2, with the current profile."""
    from scipy.constants import c as c_light

    fig, axes = plt.subplots(figsize=(5.6, 4.6), nrows=2, ncols=2,
                             sharex="col", layout="constrained",
                             gridspec_kw={"height_ratios": [3, 1.2]})
    for col, (pa, label) in enumerate(((pa_in, "P1 (entrance)"),
                                       (pa_out, "P2 (exit)"))):
        tau, dp = pa.rparticles[4], pa.rparticles[5]
        z = -(tau - tau.mean()) * 1e3          # head to the right
        d = (dp - dp.mean()) * 100.0
        ax_p, ax_c = axes[0, col], axes[1, col]
        ax_p.hexbin(z, d, gridsize=90, bins="log", cmap="viridis", mincnt=1,
                    linewidths=0.0)
        ax_p.set_ylabel(r"$\delta$ [%]")
        ax_p.set_title(f"{label}: $\\sigma_z$ = {tau.std()*1e3:.3f} mm, "
                       f"$\\sigma_\\delta$ = {dp.std()*100:.2f} %", fontsize=9)
        edges = np.linspace(np.percentile(z, 0.5), np.percentile(z, 99.5), nbin + 1)
        h, _ = np.histogram(z, bins=edges, weights=np.abs(pa.q_array))
        cur = h * c_light / ((edges[1] - edges[0]) * 1e-3)
        ax_c.fill_between(0.5 * (edges[1:] + edges[:-1]), cur, color=C_Z,
                          alpha=0.6, lw=0)
        ax_c.set_ylabel(r"$I$ [$A$]")
        ax_c.set_ylim(bottom=0.0)
        ax_c.set_xlabel(r"$z - \langle z \rangle$ [$mm$]  (head $\to$)")
    return fig


def main():
    dist = read_astra_distribution(str(P1_DIST))
    energy_gev = beam_energy_gev(dist)
    with contextlib.redirect_stdout(io.StringIO()):
        dl = solve_achromat_quads(DOGLEG, KI, energy_gev=energy_gev)
    lattice = dl.lattice
    collective = SC or CSR
    physics = (" + ".join(n for n, on in (("SC", SC), ("CSR", CSR)) if on)
               or "collective effects OFF")

    print(dl.summary())
    print(f"  beam: {P1_DIST.relative_to(REPO)}, n={len(dist)}, "
          f"Q={abs(dist.get_data('Q').sum())*1e9:.3f} nC, "
          f"E={energy_gev*1e3:.2f} MeV")
    print(f"  collective track: SC {'on' if SC else 'off'}"
          f"{' (' + 'x'.join(map(str, SC_MESH)) + ')' if SC else ''}, "
          f"CSR {'on' if CSR else 'off'}, unit_step {UNIT_STEP} m\n")

    r_in = beam_report(as_ocelot(dist))
    print("linear reference (SC and CSR off) ...")
    evo0, pa0 = track_compressor(lattice, dist, sc=False, csr=False,
                                 unit_step=UNIT_STEP)
    r_lin = beam_report(pa0)
    if collective:
        print("collective track ...")
        evo, pa = track_compressor(lattice, dist, sc=SC, csr=CSR,
                                   unit_step=UNIT_STEP, nmesh=SC_MESH,
                                   csr_nbin=CSR_NBIN)
    else:
        evo, pa = evo0, pa0
    r_out = beam_report(pa)

    P2_DIST.parent.mkdir(parents=True, exist_ok=True)
    write_astra_distribution(P2_DIST, from_ocelot_particle_array(pa))
    print(f"  wrote P2 distribution: {P2_DIST.relative_to(REPO)}\n")

    print("chirp is quoted against z (head positive); dE is the mean energy "
          "change, i.e. the CSR loss")
    print_report("P1", r_in)
    print_report("P2 linear", r_lin, r_in)
    print_report("P2 tracked", r_out, r_in)
    print(f"\n  collective effects: eps_n,x {r_lin['enx_c']*1e6:.3f} -> "
          f"{r_out['enx_c']*1e6:.3f} um ({(r_out['enx_c']/r_lin['enx_c']-1)*100:+.1f} %), "
          f"eps_n,y {r_lin['eny']*1e6:.3f} -> {r_out['eny']*1e6:.3f} um "
          f"({(r_out['eny']/r_lin['eny']-1)*100:+.1f} %),")
    print(f"  {'':2s}sigma_z {r_lin['sig_z']*1e3:.4f} -> {r_out['sig_z']*1e3:.4f} mm "
          f"({(r_out['sig_z']/r_lin['sig_z']-1)*100:+.1f} %), residual eta "
          f"{r_lin['eta']*1e3:+.2f} -> {r_out['eta']*1e3:+.2f} mm")
    print(f"  peak sizes along the dogleg: sigma_x {evo['sig_x'].max()*1e3:.2f} mm "
          f"(betatron {evo['sig_xb'].max()*1e3:.2f}), sigma_y "
          f"{evo['sig_y'].max()*1e3:.2f} mm; peak eta {evo['eta'].max()*1e3:.1f} mm")

    apply_style()
    save(figure_evolution(evo, evo0, lattice, physics), FIGS,
         "fig_OP2_dogleg_track")
    save(figure_lps(as_ocelot(dist), pa), FIGS, "fig_OP2_dogleg_lps")
    plt.show()


if __name__ == "__main__":
    main()
