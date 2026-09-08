#!/usr/bin/env python3
r"""Run one working point's middle-optics line, P0 -> P3, from the design of record.

    P0 --[M1]--> P1 --[compressor]--> P2 --[M3]--> P3

This is the SIMULATION of the line, not its design. Nothing is solved here:
every section is rebuilt from `thzim.record` -- the geometry and the selected
strengths that the per-OP scripts in this directory derived -- and the beam
is tracked through it section by section with space charge everywhere and
CSR in the compressor, exactly the physics each per-OP script used. Use the
per-OP scripts to CHANGE the design (they show why each number is what it
is); use this one to REPRODUCE its result, to start from a different input
beam (the 1M-particle distributions, a re-optimised injector), or to re-run
part of the line from a saved plane.

Every interface plane is written as an ASTRA file, so a later section can be
re-run alone (`--start p2`) and the P3 beam is the handover to the FEL
simulation, which is delivered separately.
The planes are the section boundaries as the design defines them: P1 is the
compressor's entrance marker (0.2 m before the first chicane bend; the dogleg
START marker, `d_in` before B1), P2 its exit marker (right after the last
chicane bend; `d_out` after B2 for the dogleg), P3 the undulator entrance
`d_out` = 0.4 m past the last M3 quad.

Outputs, under `outputs/<OP>/line/` (override with `--out`):

    p1.ast, p2.ast, p3.ast      the beam at each plane
    summary.json                the run settings (input, planes, SC/CSR, per-
                                section step and mesh) and a per-plane beam
                                report (sizes, Twiss, emittance, dispersion,
                                bunch length, current, energy spread) plus the
                                P3 Bmag against the target of record. Twiss and
                                Bmag are on the DISPERSION-CORRECTED moments;
                                the M3 scripts quote the projected ones, so
                                their Bmag_x reads 1.000 where this one reads
                                1.003 (OP1) / 1.010 (OP2)
    fig_<OP>_line.{png,pdf}     the whole line: sizes, emittance, dispersion,
                                compression, with the planes and magnets marked
                                (written to outputs/<OP>/figures/; a partial
                                run is suffixed `_<start>_<stop>`)

The evolution figure draws the DISPERSION-CORRECTED horizontal emittance and
the betatron size next to the total, as the compressor scripts do: inside the
bends the projected quantities swing with eta and say nothing.

Usage:

    python run_line.py OP3                          # the design, 50k P0 beam
    python run_line.py OP1 --input path/to/OP1_1M.dist --out outputs/OP1/line_1M
    python run_line.py OP4 --start p2               # M3 only, from line/p2.ast
    python run_line.py OP2 --no-sc --no-csr         # the linear reference
    python run_line.py OP3 --sc-mesh 31 --unit-step 0.05 --no-figure
                                                    # override every section

Needs `pip install -e .` at the repo root plus `partdist`. Runtime with the
50k beams: 3-8 minutes per OP, dominated by the SC field solve.
"""

import argparse
import json
import time
from pathlib import Path

from partdist import (from_ocelot_particle_array, read_astra_distribution,
                      write_astra_distribution)
from thzim import record
from thzim.compressor import beam_report, track_compressor
from thzim.quadruplet import bmag
from thzim.triplet import as_ocelot, beam_energy_gev
from thzim.utils import apply_style, output_dir, save

REPO = Path(__file__).resolve().parents[1]
C_X, C_Y, C_Z = "tab:blue", "tab:red", "tab:green"   # x blue, y red, z green


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Track one operating point P0 -> P3 from the design of record.")
    p.add_argument("op", help="operating point: OP1, OP2, OP3 or OP4")
    p.add_argument("--input", help="input beam (ASTRA format). Default: the "
                   "canonical data/<OP>_50k.dist when starting at p0, else "
                   "<out>/<start>.ast")
    p.add_argument("--start", default="p0", choices=record.PLANES[:-1],
                   help="plane to start from (default p0)")
    p.add_argument("--stop", default="p3", choices=record.PLANES[1:],
                   help="plane to stop at (default p3)")
    p.add_argument("--out", help="output directory (default outputs/<OP>/line)")
    p.add_argument("--no-sc", action="store_true", help="space charge off")
    p.add_argument("--no-csr", action="store_true",
                   help="CSR off in the compressor")
    p.add_argument("--unit-step", type=float, default=None,
                   help="navigator step [m] for EVERY section; default: each "
                        "section's own value from the record")
    p.add_argument("--sc-mesh", type=int, default=None,
                   help="SC mesh cells per axis for EVERY section; default: "
                        "each section's own value from the record")
    p.add_argument("--csr-nbin", type=int, default=300,
                   help="CSR longitudinal bins (default 300)")
    p.add_argument("--no-figure", action="store_true",
                   help="skip the whole-line evolution figure")
    return p.parse_args(argv)


# --------------------------------- reporting ---------------------------------

def _portable(path):
    """A path as recorded in summary.json: relative to the repo when inside it."""
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def plane_report(pa, target=None):
    """`beam_report` plus, when a target is given, the Bmag against it."""
    r = beam_report(pa)
    r = {k: float(v) for k, v in r.items()}
    if target is not None:
        r["bmag_x"] = float(bmag(r["beta_x"], r["alpha_x"], target[0], target[1]))
        r["bmag_y"] = float(bmag(r["beta_y"], r["alpha_y"], target[2], target[3]))
    return r


def print_plane(label, r):
    print(f"  {label:<3s} sig_z={r['sig_z']*1e3:.4f} mm  I_pk={r['I_peak']:.0f} A  "
          f"sig_dp={r['sig_dp']*100:.3f} %  enx={r['enx']*1e6:.3f} "
          f"({r['enx_c']*1e6:.3f} corr) um  eny={r['eny']*1e6:.3f} um  "
          f"eta={r['eta']*1e3:+.2f} mm")
    line = (f"  {'':3s} bx={r['beta_x']:.3f} ax={r['alpha_x']:+.3f}  "
            f"by={r['beta_y']:.4f} ay={r['alpha_y']:+.3f}")
    if "bmag_x" in r:
        line += f"   Bmag={r['bmag_x']:.3f}/{r['bmag_y']:.3f} against the P3 target"
    print(line)


# ---------------------------------- figure ----------------------------------

def figure_line(rec, runs, physics):
    """The whole line, tracked: one column of panels against s."""
    import matplotlib.pyplot as plt

    fig, (ax_s, ax_e, ax_d, ax_z) = plt.subplots(
        figsize=(6.4, 8.6), nrows=4, sharex=True, layout="constrained")
    s_off, ticks, labels = 0.0, [0.0], [runs[0]["segment"].plane_in.upper()]
    for run in runs:
        seg, evo = run["segment"], run["evo"]
        s = evo["s"] + s_off
        for ax in (ax_s, ax_e, ax_d, ax_z):
            for a, b, kind in seg.spans():
                ax.axvspan(s_off + a, s_off + b,
                           color="#c8c8c8" if kind == "bend" else "#8ecae6",
                           alpha=0.40, lw=0, zorder=0)
        ax_s.plot(s, evo["sig_x"] * 1e3, color=C_X, lw=1.4, ls="-")
        ax_s.plot(s, evo["sig_xb"] * 1e3, color=C_X, lw=1.0, ls="--")
        ax_s.plot(s, evo["sig_y"] * 1e3, color=C_Y, lw=1.4, ls="-")
        ax_e.plot(s, evo["enx_c"] * 1e6, color=C_X, lw=1.4, ls="-")
        ax_e.plot(s, evo["eny"] * 1e6, color=C_Y, lw=1.4, ls="-")
        ax_d.plot(s, evo["eta"] * 1e3, color=C_X, lw=1.4, ls="-")
        ax_z.plot(s, evo["sig_z"] * 1e3, color=C_Z, lw=1.4, ls="-")
        s_off += evo["s"][-1]
        ticks.append(s_off)
        labels.append(seg.plane_out.upper())
    ax_i = ax_z.twinx()
    s_off = 0.0
    for run in runs:
        evo = run["evo"]
        ax_i.plot(evo["s"] + s_off, evo["I_peak"], color="k", lw=1.0, ls="-")
        s_off += evo["s"][-1]
    for ax in (ax_s, ax_e, ax_d, ax_z):
        for t in ticks[1:-1]:
            ax.axvline(t, color="0.3", lw=0.8, ls=":")
    ax_s.plot([], [], color=C_X, lw=1.4, ls="-", label=r"$\sigma_x$ total")
    ax_s.plot([], [], color=C_X, lw=1.0, ls="--", label=r"$\sigma_x$ betatron")
    ax_s.plot([], [], color=C_Y, lw=1.4, ls="-", label=r"$\sigma_y$")
    ax_s.set_ylabel(r"$\sigma$ [$mm$]")
    ax_s.set_ylim(bottom=0.0)
    ax_s.legend(loc="upper left", fontsize=7, framealpha=0.85)
    ax_s.set_title(f"(a) rms size -- {rec.name}, {rec.branch}, "
                   f"{rec.charge_nc:g} nC, {rec.ekin_mev:g} MeV "
                   f"({physics})",
                   fontsize=10)
    ax_e.plot([], [], color=C_X, lw=1.4, ls="-",
              label=r"$\epsilon_{n,x}$ (dispersion-corrected)")
    ax_e.plot([], [], color=C_Y, lw=1.4, ls="-", label=r"$\epsilon_{n,y}$")
    ax_e.set_ylabel(r"$\epsilon_n$ [$\mu m$]")
    ax_e.legend(loc="upper left", fontsize=7, framealpha=0.85)
    ax_e.set_title("(b) normalised emittance", fontsize=10)
    ax_d.axhline(0.0, color="0.5", lw=0.6)
    ax_d.set_ylabel(r"$\eta_x$ [$mm$]")
    ax_d.set_title("(c) dispersion (statistical)", fontsize=10)
    ax_z.set_ylabel(r"$\sigma_z$ [$mm$]")
    ax_z.set_ylim(bottom=0.0)
    ax_i.set_ylabel(r"$I_{peak}$ [$A$]")
    ax_i.set_ylim(bottom=0.0)
    ax_z.plot([], [], color=C_Z, lw=1.4, ls="-", label=r"$\sigma_z$")
    ax_z.plot([], [], color="k", lw=1.0, ls="-", label=r"$I_{peak}$")
    ax_z.legend(loc="center left", fontsize=7, framealpha=0.85)
    ax_z.set_title("(d) bunch length and peak current", fontsize=10)
    ax_z.set_xlim(0.0, ticks[-1])
    ax_z.set_xticks(ticks, labels)
    ax_z.set_xlabel(r"$s$ along the line, interface planes marked")
    return fig


# ----------------------------------- main -----------------------------------

def main(argv=None):
    args = parse_args(argv)
    rec = record.get(args.op)
    out = Path(args.out) if args.out else output_dir(REPO, rec.name, "line")
    out.mkdir(parents=True, exist_ok=True)
    figs = output_dir(REPO, rec.name, "figures")
    sc = not args.no_sc

    i0, i1 = record.PLANES.index(args.start), record.PLANES.index(args.stop)
    if i1 <= i0:
        raise SystemExit(f"--stop {args.stop} is not downstream of --start {args.start}")
    segments = [seg for seg in rec.segments
                if record.PLANES.index(seg.plane_in) >= i0
                and record.PLANES.index(seg.plane_out) <= i1]

    if args.input:
        src = Path(args.input)
    elif args.start == "p0":
        src = REPO / "data" / f"{rec.name}_50k.dist"
    else:
        src = out / f"{args.start}.ast"
    dist = read_astra_distribution(str(src))
    energy_gev = beam_energy_gev(dist)

    print(rec.summary())
    print(f"  beam E_total {energy_gev*1e3:.3f} MeV, nominal {rec.energy_gev*1e3:.3f} MeV"
          + (" (the beam's converts the two-triplet gradients to k1, as the e2e "
             "script did)" if rec.branch == "SASE" else ""))
    print(f"  input: {src}  n={len(dist)}  "
          f"Q={abs(dist.get_data('Q').sum())*1e9:.3f} nC")
    print(f"  running {args.start} -> {args.stop}: "
          + ", ".join(seg.name for seg in segments))
    override = ""
    if args.unit_step is not None or args.sc_mesh is not None:
        override = (" -- OVERRIDDEN to "
                    + (f"step {args.unit_step} m" if args.unit_step else "")
                    + (", " if args.unit_step and args.sc_mesh else "")
                    + (f"mesh {args.sc_mesh}^3" if args.sc_mesh else "")
                    + " in every section")
    print(f"  SC {'on' if sc else 'off'}, CSR in the compressor "
          f"{'off' if args.no_csr else 'on'}; tracking settings per section "
          f"as recorded{override}; output {out}\n")

    summary = {"op": rec.name, "input": _portable(src), "start": args.start,
               "stop": args.stop, "sc": sc, "csr": not args.no_csr,
               "sections": {}, "planes": {}}
    r0 = plane_report(as_ocelot(dist))
    summary["planes"][args.start] = r0
    print_plane(args.start.upper(), r0)

    runs = []
    for seg in segments:
        lattice = seg.lattice(energy_gev)
        csr = seg.csr and not args.no_csr
        unit_step = args.unit_step or seg.unit_step
        mesh = (args.sc_mesh,) * 3 if args.sc_mesh else seg.sc_mesh
        summary["sections"][seg.name] = {"unit_step": unit_step,
                                         "sc_mesh": list(mesh), "csr": csr}
        t0 = time.time()
        evo, pa = track_compressor(lattice, dist, sc=sc, csr=csr,
                                   unit_step=unit_step, nmesh=mesh,
                                   csr_nbin=args.csr_nbin)
        dist = from_ocelot_particle_array(pa)
        path = out / f"{seg.plane_out}.ast"
        write_astra_distribution(path, dist)
        target = rec.p3_target if seg.plane_out == "p3" else None
        r = plane_report(pa, target)
        summary["planes"][seg.plane_out] = r
        runs.append(dict(segment=seg, evo=evo))
        print(f"\n{seg.name} ({lattice.totalLen:.3f} m, SC "
              f"{'on ' + 'x'.join(map(str, mesh)) if sc else 'off'}, "
              f"CSR {'on' if csr else 'off'}, step {unit_step:g} m): "
              f"{time.time() - t0:.0f} s, "
              f"wrote {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
        print(f"  peak sigma_x {evo['sig_x'].max()*1e3:.2f} mm "
              f"(betatron {evo['sig_xb'].max()*1e3:.2f}), "
              f"sigma_y {evo['sig_y'].max()*1e3:.2f} mm")
        print_plane(seg.plane_out.upper(), r)

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out / 'summary.json'}")

    if not args.no_figure and runs:
        apply_style()
        tag = "" if (args.start, args.stop) == ("p0", "p3") else f"_{args.start}_{args.stop}"
        physics = (" + ".join(n for n, on in (("SC", sc), ("CSR", not args.no_csr)) if on)
                   or "collective effects OFF")
        save(figure_line(rec, runs, physics), figs,
             f"fig_{rec.name}_line{tag}")
        print(f"wrote {figs / f'fig_{rec.name}_line{tag}.png'}")


if __name__ == "__main__":
    main()
