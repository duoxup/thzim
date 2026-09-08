#!/usr/bin/env python3
r"""OP3 chicane — the horizontal entrance beta: a tracking scan.

The chicane's entrance Twiss splits into a hard solve and a design choice:

  y   an edge-focusing channel with a periodic solution. `thzim.chicane.
      matched_y` gives (beta*, alpha*) analytically -- no scan, no distribution.
      Injecting it keeps beta_y flat, so no high-density waist forms, so the
      linear criterion and the collective one agree.
  x   EXACTLY a drift (a rectangular bend is horizontally a drift of its chord),
      so there is no equilibrium optics and beta_x is FREE. Nothing pins it in
      linear theory; it is settled by space charge, which needs tracking with a
      real distribution. That is this script.

The scan does not collapse the trade-off into an artificial scalar objective.
It supports a soft engineering choice, and beta_x = 30 m is the adopted nominal
for OP3. The second figure puts the three candidate exit objectives (eps_n,x,
eps_n,y, sigma_z) in one 3-D scatter and marks which beta_x are Pareto-optimal.
The first figure shows, per entrance beta_x:

  (a) beta_x(s), against the collective-free drift curve beta0 + s^2/beta0
  (b) sigma_x(s), total and dispersion-corrected
  (c) eps_n,x(s) dispersion-corrected, with eps_n,y for comparison -- where
      along the chicane the growth actually happens. The PROJECTED eps_n,x is
      deliberately not drawn: inside the chicane eta reaches ~0.26 m and, with
      sigma_delta ~0.5 %, the dispersive term takes it to ~100 um and back
      again. That is bookkeeping, not growth. Only its exit value means
      anything, and the table prints it.
  (d) sigma_z(s), against the same run with SC and CSR switched off -- how much
      say beta_x has over the bunching at all. Two separate things show up here.
      The mid-chicane spread is NOT compression: it is the z-x coupling R51,
      R52, the symplectic transpose partner of the dispersion, which adds
      |R51| sigma_x,beta of path-length spread (0.09 / 0.12 / 0.28 mm at
      beta_x = 1 / 10 / 60 m, matching what the panel shows) and then cancels
      EXACTLY at the exit, because the full chicane has R51 = R52 = 0 by
      symmetry. Strip the collective effects and every beta_x reaches the same
      exit sigma_z. What is left is the real effect: SC and CSR spoil the
      compression by ~8 % at beta_x = 1 m but only ~5 % at 60 m, so a larger
      beam bunches slightly better too.

Read (a) with care. The plotted beta_x is STATISTICAL, sigma_xbeta^2 / eps_x,
and eps_x grows under space charge, so the curve SAGS below the drift reference
even though the physical betatron size does not shrink. The beam is not being
focused; the denominator is moving. That is why (b) is the panel to judge
optics by. The two panels together are the point of this figure.

Note also that the beta_x minimising the beam SIZE is roughly L_x (the
equivalent x-drift length, ~2.17 m here) -- and that is the WORST choice for
emittance, because it maximises density. The design therefore runs deliberately
in the opposite direction, large beta_x, until the betatron size overtakes the
dispersive floor eta sigma_delta and the emittance benefit saturates.

Input: `data/OP3_50k.dist`, conditioned per scan point with
`partdist.pd3d.manipulator.match_twiss_xy` -- x to (alpha=0, beta_x), y to the
matched solution. Emittance and the longitudinal phase space are untouched
(Courant-Snyder is symplectic in the plane), so every point starts from the same
beam and differs only in the transverse optics.

SC mesh: 63^3 is the default and costs little -- the runtime is dominated by
depositing/interpolating the particles, not by the FFT, so 63^3 is only ~15 %
slower than 31^3. Measured convergence at beta_x = 30 m with SC+CSR, 50k
particles:

    mesh       t/s    enx proj    enx corr     eny     exit eta
    15x15x15   11.4    2.4974 um   2.4894 um   1.2040   11.65 mm
    31x31x31   12.9    2.5747      2.5654      1.2096   12.24
    63x63x63   14.9    2.6225      2.6129      1.2122   12.37

Beam SIZE is converged already at 15^3 (peak sigma_x agrees to 4 digits); the
emittance and the residual dispersion are not, and still creep ~2 % from 31^3 to
63^3. Repeat the convergence check if the mesh or tracking model changes.

Run:  python chicane_beta_x_scan.py   (needs `pip install -e .` at the repo root
      plus `partdist`; ~15 s per scan point)
"""

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import ocelot as oc
from ocelot.adaptors.astra2ocelot import astraBeam2particleArray
from ocelot.cpbd.csr import CSR
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.physics_proc import PhysProc
from ocelot.cpbd.sc import SpaceCharge
from ocelot.cpbd.track import track
from partdist import read_astra_distribution, write_astra_distribution
from partdist.pd3d.manipulator import match_twiss_xy

from thzim.chicane import ChicaneGeom, chicane_seq, element_spans, matched_y
from thzim.utils import apply_style, output_dir, save

# --------------------------- user settings ---------------------------
REPO = Path(__file__).resolve().parents[2]
DIST = REPO / "data" / "OP3_50k.dist"
FIGS = output_dir(REPO, "OP3", "figures")

GEOM = ChicaneGeom(theta_deg=16.69, l_bz=0.10, l_dz=0.75, lead=0.20)
EKIN_MEV = 39.9                       # OP3
BETA_X_SCAN = [1.0, 3.0, 10.0, 30.0, 60.0]     # entrance beta_x [m], alpha_x = 0
BETA_X_NOMINAL = 30.0                 # adopted value; selected from this scan
ALPHA_X = 0.0                         # waist at the entrance; see the docstring

SC_MESH = [63, 63, 63]                # None to switch space charge off
USE_CSR = True
UNIT_STEP = 0.02                      # navigator step [m]
CSR_NBIN = 300
# ---------------------------------------------------------------------

M_E_MEV = 0.51099895


class TwissMonitor(PhysProc):
    """Records the transverse moments along s, separating the dispersive part."""

    def __init__(self):
        super().__init__()
        self.rec = []

    def apply(self, p, dz):
        x, xp, y, yp, tau, dp = p.rparticles
        d = dp - dp.mean()
        var_d = d.var()
        eta = float(np.mean((x - x.mean()) * d) / var_d) if var_d > 0 else 0.0
        etap = float(np.mean((xp - xp.mean()) * d) / var_d) if var_d > 0 else 0.0
        xb, xpb = (x - x.mean()) - eta * d, (xp - xp.mean()) - etap * d
        eps = _emit(xb, xpb)
        gamma = p.E / (M_E_MEV * 1e-3)
        self.rec.append(dict(
            s=float(p.s), sig_x=float(x.std()), sig_xb=float(xb.std()),
            sig_y=float(y.std()), sig_z=float(tau.std()), eta=eta, eps_x=eps,
            beta_x=float(xb.var() / eps) if eps > 0 else np.nan,
            enx=gamma * _emit(x, xp), enx_c=gamma * eps,
            eny=gamma * _emit(y, yp)))


def _emit(u, up):
    """Geometric rms emittance of one plane."""
    cov = np.cov(np.vstack([u - u.mean(), up - up.mean()]))
    return float(np.sqrt(max(np.linalg.det(cov), 0.0)))


def emit_n(p, plane="x", disp_corrected=False):
    """Normalised projected emittance [m rad]."""
    x, xp, y, yp, tau, dp = p.rparticles
    u, up = (x, xp) if plane == "x" else (y, yp)
    if disp_corrected:
        d = dp - dp.mean()
        var_d = d.var()
        eta = np.mean((u - u.mean()) * d) / var_d
        etap = np.mean((up - up.mean()) * d) / var_d
        u, up = (u - u.mean()) - eta * d, (up - up.mean()) - etap * d
    return p.E / (M_E_MEV * 1e-3) * _emit(u, up)


def condition(dist, beta_x, alpha_x, beta_y, alpha_y, workdir):
    """Set the entrance Twiss with partdist, then hand the beam to Ocelot.

    The round trip goes through an ASTRA file on purpose: it reuses Ocelot's own
    adaptor rather than re-deriving the momentum-deviation convention here.
    """
    matched = match_twiss_xy(dist, alpha_x=alpha_x, beta_x=beta_x,
                             alpha_y=alpha_y, beta_y=beta_y)
    path = Path(workdir) / f"op3_bx{beta_x:g}.ast"
    write_astra_distribution(path, matched)
    with contextlib.redirect_stdout(io.StringIO()):     # adaptor is chatty
        return astraBeam2particleArray(str(path))


def track_chicane(p, geom, collective=True):
    """Track through the chicane, returning the monitor record.

    collective=False switches SC and CSR off, which costs ~0.1 s and gives
    the linear-optics reference the plots compare against.
    """
    seq, _ = chicane_seq(geom)
    lat = oc.MagneticLattice(seq)
    navi = Navigator(lat)
    navi.unit_step = UNIT_STEP
    if USE_CSR and collective:
        csr = CSR()
        csr.n_bin = CSR_NBIN
        navi.add_physics_proc(csr, seq[0], seq[-1])
    if SC_MESH and collective:
        sc = SpaceCharge()
        sc.nmesh_xyz = list(SC_MESH)
        sc.step = 1
        navi.add_physics_proc(sc, seq[0], seq[-1])
    mon = TwissMonitor()
    navi.add_physics_proc(mon, seq[0], seq[-1])
    with contextlib.redirect_stdout(io.StringIO()):
        track(lat, p, navi, print_progress=False, calc_tws=False)
    return {k: np.array([r[k] for r in mon.rec]) for k in mon.rec[0]}


def main():
    energy_gev = (EKIN_MEV + M_E_MEV) * 1e-3
    my = matched_y(GEOM, energy_gev)
    if my is None:
        raise SystemExit(f"y channel unstable at theta = {GEOM.theta_deg} deg")
    beta_y, alpha_y, mu_y = my

    print(GEOM.summary())
    print(f"  matched y: beta*={beta_y:.4f} m  alpha*={alpha_y:+.4f}  "
          f"mu={mu_y:.2f} deg   (held fixed across the scan)")
    print(f"  adopted horizontal target: beta_x={BETA_X_NOMINAL:g} m, "
          f"alpha_x={ALPHA_X:+g}")
    dist = read_astra_distribution(str(DIST))
    print(f"  beam: {DIST.name}, n={len(dist)}, "
          f"Q={abs(dist.get_data('Q').sum())*1e9:.3f} nC, "
          f"sigma_z={dist.std('z')*1e3:.4f} mm, "
          f"sigma_pz/pz0={dist.std('pz')/dist.mean('pz')*100:.4f} %")
    mesh = "off" if not SC_MESH else "x".join(map(str, SC_MESH))
    ncell = 1 if not SC_MESH else int(np.prod(SC_MESH))
    print(f"  SC mesh {mesh} ({len(dist)/ncell:.2f} particles/cell if the box were "
          f"full -- a poor guide, see the docstring's convergence table), "
          f"CSR {'on' if USE_CSR else 'off'}, unit_step {UNIT_STEP} m\n")

    runs = []
    with tempfile.TemporaryDirectory() as workdir:
        for beta_x in BETA_X_SCAN:
            p = condition(dist, beta_x, ALPHA_X, beta_y, alpha_y, workdir)
            evo = track_chicane(p, GEOM)
            evo["s"] = evo["s"] - evo["s"][0]
            p0 = condition(dist, beta_x, ALPHA_X, beta_y, alpha_y, workdir)
            evo0 = track_chicane(p0, GEOM, collective=False)
            evo0["s"] = evo0["s"] - evo0["s"][0]
            runs.append(dict(
                beta_x=beta_x, evo=evo, evo0=evo0,
                sig_z_out0=evo0["sig_z"][-1],
                enx=emit_n(p, "x") * 1e6,
                enx_c=emit_n(p, "x", disp_corrected=True) * 1e6,
                eny=emit_n(p, "y") * 1e6,
                peak_sig_x=evo["sig_x"].max(), peak_sig_xb=evo["sig_xb"].max(),
                eta_exit=evo["eta"][-1],
                sig_z_in=evo["sig_z"][0], sig_z_out=evo["sig_z"][-1]))
            print(f"  beta_x = {beta_x:5.1f} m done")

    print(f"\n{'beta_x/m':>9s} {'peak sig_x/mm':>14s} {'peak sig_xb/mm':>15s} "
          f"{'enx/um':>8s} {'enx corr/um':>12s} {'eny/um':>8s} "
          f"{'exit eta/mm':>12s} {'sig_z out/mm':>13s} {'C':>7s} "
          f"{'vs linear':>10s}")
    for r in runs:
        print(f"{r['beta_x']:9.1f} {r['peak_sig_x']*1e3:14.4f} "
              f"{r['peak_sig_xb']*1e3:15.4f} {r['enx']:8.4f} {r['enx_c']:12.4f} "
              f"{r['eny']:8.4f} {r['eta_exit']*1e3:12.2f} "
              f"{r['sig_z_out']*1e3:13.5f} "
              f"{r['sig_z_in']/r['sig_z_out']:7.3f} "
              f"{(r['sig_z_out']/r['sig_z_out0']-1)*100:+9.2f} %")
    spread = (max(r['sig_z_out'] for r in runs)
              / min(r['sig_z_out'] for r in runs) - 1.0)
    print(f"exit sigma_z varies by {spread*100:.2f} % across the beta_x scan "
          f"(sigma_z in = {runs[0]['sig_z_in']*1e3:.4f} mm). Without collective "
          f"effects it is\n{'':2s}{runs[0]['sig_z_out0']*1e3:.5f} mm for EVERY "
          f"beta_x -- the compression itself is optics, so the last column is "
          f"the whole\n{'':2s}effect beta_x has on bunching: a larger beam "
          f"loses less of it to space charge and CSR.")
    print(f"\nbeam SIZE is minimised near beta_x ~ L_x = {GEOM.L_x:.2f} m, "
          f"which is the WORST point for emittance -- see the docstring.")

    # ------------------------------ plot ------------------------------
    apply_style()
    fig, (ax_b, ax_s, ax_e, ax_z) = plt.subplots(figsize=(5.5, 9.6), nrows=4,
                                                 sharex=True,
                                                 layout="constrained")
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(runs)))

    for ax in (ax_b, ax_s, ax_e, ax_z):
        for s0, s1, _ in element_spans(GEOM):
            ax.axvspan(s0, s1, color="#c8c8c8", alpha=0.35, lw=0)
    ax_z.set_xlim(0.0, GEOM.length)
    ax_z.set_xlabel(r"$s$ [$m$]")

    # linestyle is explicit everywhere: the scienceplots "ieee" style cycles
    # linestyles as well as colours, which would otherwise override the
    # solid/dashed convention these panels rely on
    for r, c in zip(runs, colors):
        s, b0 = r["evo"]["s"], r["beta_x"]
        ax_b.plot(s, r["evo"]["beta_x"], color=c, lw=1.5, ls="-",
                  label=rf"$\beta_x$ = {b0:g} m")
        ax_b.plot(s, b0 + s**2 / b0, color=c, lw=0.9, ls=":")
        ax_s.plot(s, r["evo"]["sig_x"] * 1e3, color=c, lw=1.5, ls="-")
        ax_s.plot(s, r["evo"]["sig_xb"] * 1e3, color=c, lw=1.2, ls="--")
        ax_e.plot(s, r["evo"]["enx_c"] * 1e6, color=c, lw=1.5, ls="-")
        ax_e.plot(s, r["evo"]["eny"] * 1e6, color=c, lw=1.2, ls="--")
        ax_z.plot(s, r["evo"]["sig_z"] * 1e3, color=c, lw=1.5, ls="-")
        ax_z.plot(s, r["evo0"]["sig_z"] * 1e3, color=c, lw=0.9, ls=":")

    ax_b.set_yscale("log")
    ax_b.set_ylabel(r"$\beta_x$ [$m$] (statistical)")
    ax_b.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.85)
    ax_b.set_title(r"(a) tracked $\beta_x$ vs the drift curve "
                   r"$\beta_0 + s^2/\beta_0$ (dotted)", fontsize=10)

    ax_s.set_ylabel(r"$\sigma_x$ [$mm$]")
    ax_s.plot([], [], color="0.3", lw=1.5, ls="-", label="total")
    ax_s.plot([], [], color="0.3", lw=1.2, ls="--", label="dispersion-corrected")
    ax_s.legend(loc="upper left", fontsize=7, framealpha=0.85)
    ax_s.set_title("(b) beam size: the dispersive floor and the betatron part",
                   fontsize=10)

    # Only the DISPERSION-CORRECTED eps_x is drawn. The projected one is
    # meaningless inside the chicane -- with eta up to ~0.26 m and
    # sigma_delta ~0.5 %, the dispersive term drives it to ~100 um and back,
    # which is bookkeeping, not growth. Its exit value (where eta -> 0, so the
    # two agree) is in the printed table.
    ax_e.axhline(runs[0]["evo"]["enx_c"][0] * 1e6, color="0.4", lw=0.9, ls=":")
    ax_e.set_ylabel(r"$\epsilon_n$ [$\mu m$]")
    ax_e.plot([], [], color="0.3", lw=1.5, ls="-",
              label=r"$\epsilon_{n,x}$ (dispersion-corrected)")
    ax_e.plot([], [], color="0.3", lw=1.2, ls="--", label=r"$\epsilon_{n,y}$")
    ax_e.plot([], [], color="0.4", lw=0.9, ls=":", label="entrance value")
    ax_e.legend(loc="upper left", fontsize=7, framealpha=0.85)
    ax_e.set_title("(c) where the emittance growth happens", fontsize=10)

    ax_z.set_ylabel(r"$\sigma_z$ [$mm$]")
    ax_z.plot([], [], color="0.3", lw=1.5, ls="-", label="SC + CSR")
    ax_z.plot([], [], color="0.3", lw=0.9, ls=":", label="linear optics only")
    ax_z.legend(loc="lower left", fontsize=7, framealpha=0.85)
    ax_z.set_title(rf"(d) bunching: exit $\sigma_z$ = "
                   rf"{runs[0]['sig_z_out0']*1e3:.4f} mm without collective "
                   rf"effects, for every $\beta_x$", fontsize=10)

    # ------------------- objective space (exit values) -------------------
    # A one-parameter family, so the points must lie on a CURVE in the three
    # objectives; the line makes that explicit and the wall shadows give the
    # depth cues a 5-point 3-D scatter otherwise lacks.
    obj = np.array([[r["enx_c"], r["eny"], r["sig_z_out"] * 1e3] for r in runs])
    pareto = [i for i in range(len(obj))
              if not any(np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i])
                         for j in range(len(obj)))]
    print("\nPareto-optimal beta_x (minimising all three exit objectives): "
          + ", ".join(f"{runs[i]['beta_x']:g} m" for i in pareto))
    print("  everything else is dominated: eps_n,y and sigma_z improve "
          "monotonically with beta_x,\n  but eps_n,x turns around, so the "
          "front is the stretch past its minimum.")

    fig2 = plt.figure(figsize=(6.0, 4.8), layout="constrained")
    ax3 = fig2.add_subplot(projection="3d")
    lims = [(v.min() - 0.06 * np.ptp(v), v.max() + 0.06 * np.ptp(v))
            for v in obj.T]

    ax3.plot(obj[:, 0], obj[:, 1], obj[:, 2], color="0.6", lw=1.0, ls="-",
             zorder=1)
    for (ex, ey, sz), c, r in zip(obj, colors, runs):
        # shadows on the three walls
        ax3.plot([ex], [ey], [lims[2][0]], "o", color=c, ms=3, alpha=0.35)
        ax3.plot([ex], [lims[1][1]], [sz], "o", color=c, ms=3, alpha=0.35)
        ax3.plot([lims[0][1]], [ey], [sz], "o", color=c, ms=3, alpha=0.35)
        ax3.plot([ex], [ey], [sz], "o", color=c, ms=8, mec="black", mew=1.0,
                 zorder=3)
        ax3.text(ex, ey, sz, f"  {r['beta_x']:g}", fontsize=8, zorder=4)
    for i in pareto:
        ax3.plot([obj[i, 0]], [obj[i, 1]], [obj[i, 2]], "o", ms=14,
                 mfc="none", mec="crimson", mew=1.4, zorder=2)

    ax3.set_xlim(*lims[0])
    ax3.set_ylim(*lims[1])
    ax3.set_zlim(*lims[2])
    # the science/ieee style's minor ticks turn a 3-D box into a wire mess
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_major_locator(MaxNLocator(4))
        axis.set_minor_locator(NullLocator())
    ax3.set_xlabel(r"$\epsilon_{n,x}$ [$\mu m$]", fontsize=9, labelpad=2)
    ax3.set_ylabel(r"$\epsilon_{n,y}$ [$\mu m$]", fontsize=9, labelpad=2)
    ax3.set_zlabel(r"$\sigma_z$ [$mm$]", fontsize=9, labelpad=6)
    ax3.tick_params(labelsize=7)
    ax3.set_title("exit objectives, labelled by "
                  r"$\beta_x$ [m]; red = Pareto-optimal" "\n"
                  r"($\epsilon_{n,x}$ dispersion-corrected)", fontsize=10)
    ax3.view_init(elev=22, azim=-58)

    save(fig, FIGS, 'fig_OP3_chicane_beta_x_scan')
    save(fig2, FIGS, 'fig_OP3_chicane_beta_x_objectives')
    plt.show()


if __name__ == "__main__":
    main()
