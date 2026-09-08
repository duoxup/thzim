#!/usr/bin/env python3
r"""OP4 chicane: track the matched P1 beam through the compressor to P2.

    P1 (chicane entrance)  lead  B1  d  B2  l2/2 | C | l2/2  B3  d  B4   P2

The lattice is the OP4 design of record (`ChicaneGeom` below, the same
instance `m1_quadruplet_match.py` computed its y target from), so nothing is
solved here: the beam written by that script is read back and tracked with
space charge and CSR, and the result is what the final match (M3) will see.

Two tracks are run on the same beam and drawn together:

  * the LINEAR reference -- SC and CSR off. The compression itself is optics
    (R56 times the chirp), so this is the bunch length the lattice delivers on
    its own, and the residual dispersion the achromat leaves by symmetry.
  * the COLLECTIVE track -- SC + CSR on. The difference is what the charge
    costs: the emittance it adds, the dispersion it leaves unclosed, the
    energy it radiates away, and how much of the compression it spoils.

Figure 1, five panels against s (x blue, y red; dotted = linear reference):

  (a) rms size: the total sigma_x and its BETATRON part, dispersion removed.
      Inside the chicane the dispersive term dominates (eta reaches ~0.34 m),
      so the total says nothing about the optics; the betatron size is what
      the space charge acts on.
  (b) the statistical dispersion eta(s), closing to its exit residual.
  (c) normalised emittance: eps_n,x DISPERSION-CORRECTED, and eps_n,y. The
      projected eps_n,x is not drawn -- it swings to ~70 um and back with eta,
      which is bookkeeping, not growth; its exit value is in the table.
  (d) the compression: sigma_z on the left axis, peak current on the right.
  (e) the lattice, drawn by ocelot.

Figure 2 is the longitudinal phase space at P1 and P2 with the current profile
under each, the direct picture of what the compressor did to the chirp.

Input:  `outputs/OP4/beams/p1.ast` (from `m1_quadruplet_match.py`, SC on)
Output: `outputs/OP4/beams/p2.ast`, consumed by the final match.

Run:  python chicane_track.py     (needs `pip install -e .` at the repo root
      plus `partdist`; ~20 s with SC + CSR at 50k particles)
Prev: m1_quadruplet_match.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from partdist import (from_ocelot_particle_array, read_astra_distribution,
                      write_astra_distribution)
from thzim.chicane import ChicaneGeom, build_lattice, element_spans, r56_z_exact
from thzim.compressor import beam_report, track_compressor
from thzim.triplet import as_ocelot, beam_energy_gev
from thzim.utils import apply_style, output_dir, save

# --------------------------- user settings ---------------------------
REPO = Path(__file__).resolve().parents[2]
P1_DIST = output_dir(REPO, "OP4", "beams") / "p1.ast"
P2_DIST = output_dir(REPO, "OP4", "beams") / "p2.ast"
FIGS = output_dir(REPO, "OP4", "figures")

# OP4 design of record; must match m1_quadruplet_match.py, whose y target was
# computed from this geometry at lead = 0.2 m.
CHICANE = ChicaneGeom(theta_deg=19.05, l_bz=0.10, l_dz=0.75, lead=0.20)

SC = True                 # space charge in the collective track
CSR = True                # coherent synchrotron radiation in the collective track
SC_MESH = (63, 63, 63)    # the convention of chicane_beta_x_scan.py
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
        for a, b, _ in element_spans(CHICANE):
            ax.axvspan(a, b, color="#c8c8c8", alpha=0.35, lw=0, zorder=0)

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
    ax_l.set_xlim(-0.02 * CHICANE.length, 1.02 * CHICANE.length)
    ax_l.set_xlabel(r"$s$ [$m$]   (P1 $\to$ B1 B2 $|$ B3 B4 $\to$ P2)")
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
    lattice = build_lattice(CHICANE)
    collective = SC or CSR
    physics = (" + ".join(n for n, on in (("SC", SC), ("CSR", CSR)) if on)
               or "collective effects OFF")

    print(CHICANE.summary())
    print(f"  R56 from the Ocelot map at {energy_gev*1e3:.2f} MeV: "
          f"{r56_z_exact(CHICANE, energy_gev)*1e3:+.2f} mm (z convention, "
          f"velocity term included)")
    print(f"  dipole field {CHICANE.field(energy_gev):.4f} T")
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
    print(f"  peak sizes along the chicane: sigma_x {evo['sig_x'].max()*1e3:.2f} mm "
          f"(betatron {evo['sig_xb'].max()*1e3:.2f}), sigma_y "
          f"{evo['sig_y'].max()*1e3:.2f} mm; peak eta {evo['eta'].max()*1e3:.1f} mm")

    apply_style()
    save(figure_evolution(evo, evo0, lattice, physics), FIGS,
         "fig_OP4_chicane_track")
    save(figure_lps(as_ocelot(dist), pa), FIGS, "fig_OP4_chicane_lps")
    plt.show()


if __name__ == "__main__":
    main()
