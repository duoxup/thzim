#!/usr/bin/env python3
r"""Demo / smoke test for `thzim.triplet`: round transport through one triplet.

    marker  d0  Q1  d1  Q2  d2  Q3  d3 (exit)  marker

`solve_round_triplet` fixes the first-quad gradient g1 and solves (g2, g3) so
the beam leaves the triplet ROUND -- sigma_x = sigma_y -- and stays round across
the whole exit drift. Scanning g1 walks the one-parameter family of round
solutions; that scan is the knob the two-triplet match turns, so this demo is
the picture underneath it.

Two panels, same axes, five g1 values:

    left   SC OFF -- analytic 2x2 transport over `thzim.maps`
    right  SC ON  -- the SAME knobs, ocelot-tracked with space charge

    colour = g1,  solid = sigma_x,  dashed = sigma_y

Read them as a pair. On the left every g1 gives two curves lying exactly on top
of each other after the Q3 exit (grey band): that is the round condition, and it
holds identically because the linear solve put it there. On the right the same
lattice carries a real bunch, and the question the figure answers is how much of
that roundness survives -- the printed table gives the number, `sc dev`, as the
worst fractional |sigma_x - sigma_y| anywhere in the exit drift.

## Why the right panel re-uses the LINEAR knobs

That is the production convention, not a shortcut. In the original study only
TWO things ever ran with space charge on: the g1-scan screen sizes that locate
the crossing, and the final end-to-end validation. g2 and g3 came from the
linear solve throughout (its crossing driver never re-solved them under SC,
and a dedicated study -- retired, see MIGRATION.md -- checked that doing so
does not change the delivered Twiss). The physical reason is
visible in this figure: g1 sets the ABSOLUTE size at the screens, which space
charge inflates directly, while g2 and g3 enforce a RATIO between the planes --
and the SC kick of an already-round beam is symmetric, so it pushes both planes
the same way and leaves the ratio nearly alone.

Set `SC_KNOBS = True` to re-solve (g2, g3) under space charge as well and see
that claim tested. It costs one to a few minutes per g1 instead of ~3 seconds
(the bounded SC refinement was measured at 157 s on a hard OP1 point).

The Gaussian injection beam is built here rather than in `thzim.triplet`: the
package takes distributions, it does not manufacture them. Swap in
`partdist.read_astra_distribution(<path>)` to run the same demo on a real bunch.
Deliberately unequal x/y Twiss, because a beam that arrives round makes the
solve trivial and the demo pointless.

Run:  python demo_triplet_round_transport.py   (needs `pip install -e .` at the
      repo root plus `partdist`; ~20 s at the default settings)
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# this script tests the package, so it uses the package's own style helper
# rather than tools/plt_style.py (which exists for the tools that do NOT need
# thzim installed).
from thzim.maps import drift_map, propagate, quad_map
from thzim.triplet import TripletGeom, injection_sigma, solve_round_triplet
from thzim.utils import M_E_GEV, apply_style, output_dir, save

# --------------------------- user settings ---------------------------
REPO = Path(__file__).resolve().parents[1]
FIGS = output_dir(REPO, "tools", "figures")

GEOM = TripletGeom(lq=(0.15, 0.15, 0.15), drifts=(0.50, 0.15, 0.15, 2.10))
G1_SCAN = [-0.2, 0.2, 0.6, 1.0, 1.4]   # first-quad gradient [T/m]
G_SEED = (-2.0, 0.0)                   # (g2, g3) seed; g3 = 0 runs stage 1 first

# injection beam (unequal x/y on purpose -- see the docstring)
BETA_X, ALPHA_X = 12.0, 0.5
BETA_Y, ALPHA_Y = 8.0, -0.5
NEMIT = 5.0e-6                # normalised emittance [m rad], both planes
E_GEV = 0.0395                # total energy [GeV]
SIGMA_Z = 1.0e-3              # rms bunch length [m]
NPART = 40000
CHARGE = 1.0e-9               # bunch charge [C]
SEED = 0

SC_KNOBS = False              # True: re-solve (g2, g3) under SC too (minutes/g1)
SC_MESH = (31, 31, 31)
SC_UNIT_STEP = 0.05           # SC navigator step [m]
NSL = 60                      # slices per element for the analytic envelope
# ---------------------------------------------------------------------


def gaussian_beam(beta_x, alpha_x, beta_y, alpha_y, nemit, energy_gev,
                  sigma_z=1e-3, npart=20000, seed=0, charge=1e-9):
    """A partdist ParticleDistribution3D with the prescribed Twiss.

    Script-side by design: `thzim` consumes distributions, it does not create
    them. Pure numpy sampling, so partdist is the only import needed.
    """
    import partdist
    from partdist import kinematics as kin

    rng = np.random.default_rng(seed)
    gamma = energy_gev / M_E_GEV
    eps = nemit / np.sqrt(gamma**2 - 1.0)          # geometric emittance

    def block(beta, alpha):
        cov = eps * np.array([[beta, -alpha], [-alpha, (1.0 + alpha**2) / beta]])
        return rng.multivariate_normal([0.0, 0.0], cov, npart).T

    x, xp = block(beta_x, alpha_x)
    y, yp = block(beta_y, alpha_y)
    z = rng.normal(0.0, sigma_z, npart)
    pz = kin.p_eVc_from_gamma(gamma) * (1.0 + rng.normal(0.0, 1e-4, npart))
    return partdist.ParticleDistribution3D.from_arrays(
        x=x, y=y, z=z, px=xp * pz, py=yp * pz, pz=pz,
        t=z / 299792458.0, Q=np.full(npart, charge / npart))


def envelope_linear(geom, k, sig0_x, sig0_y, nsl=NSL):
    """sigma_x(s), sigma_y(s) [m] through triplet + exit drift, analytic 2x2.

    Built from `thzim.maps` rather than living in the package: an envelope is a
    plotting convenience, and the solver needs only the two exit conditions.
    """
    lq, d = geom.lq, geom.drifts
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


def envelope_sc(dist, geom, k):
    """sigma_x(s), sigma_y(s) [m] from an ocelot space-charge track.

    Read straight off ocelot's twiss output (sigma = sqrt(emit beta)) -- no
    envelope machinery of our own on this path.
    """
    import ocelot as oc
    from partdist import to_ocelot_particle_array

    lq, d = geom.lq, geom.drifts
    seq = [oc.Marker(), oc.Drift(l=d[0]),
           oc.Quadrupole(l=lq[0], k1=k[0]), oc.Drift(l=d[1]),
           oc.Quadrupole(l=lq[1], k1=k[1]), oc.Drift(l=d[2]),
           oc.Quadrupole(l=lq[2], k1=k[2]), oc.Drift(l=d[3]), oc.Marker()]
    lat = oc.MagneticLattice(seq)
    navi = oc.Navigator(lat, unit_step=SC_UNIT_STEP)
    navi.add_physics_proc(oc.SpaceCharge(1, nmesh_xyz=list(SC_MESH)),
                          seq[0], seq[-1])
    tws, _ = oc.track(lat, to_ocelot_particle_array(dist), navi,
                      print_progress=False)
    s = np.array([t.s for t in tws])
    sx = np.sqrt(np.array([t.emit_x * t.beta_x for t in tws]))
    sy = np.sqrt(np.array([t.emit_y * t.beta_y for t in tws]))
    return s, sx, sy


def exit_roundness(s, sx, sy, s_q3):
    """Worst fractional |sx - sy| from the Q3 exit onward -- the demo's metric."""
    m = s >= s_q3 - 1e-9
    return float(np.max(np.abs(sx[m] - sy[m]) / (0.5 * (sx[m] + sy[m]))))


def main():
    print(GEOM.summary())
    print(f"  injection: beta=({BETA_X}, {BETA_Y}) m  alpha=({ALPHA_X}, "
          f"{ALPHA_Y})  eps_n={NEMIT*1e6:.1f} um  E={E_GEV*1e3:.1f} MeV  "
          f"Q={CHARGE*1e9:.2f} nC  n={NPART}")
    print(f"  SC: mesh {'x'.join(map(str, SC_MESH))}, unit_step {SC_UNIT_STEP} m"
          f"   |   (g2, g3) solved with SC: {SC_KNOBS}\n")

    dist = gaussian_beam(BETA_X, ALPHA_X, BETA_Y, ALPHA_Y, NEMIT, E_GEV,
                         sigma_z=SIGMA_Z, npart=NPART, seed=SEED, charge=CHARGE)
    sig0_x, sig0_y = injection_sigma(dist)

    runs = []
    for g1 in G1_SCAN:
        r = solve_round_triplet(dist, GEOM, g1, sc=SC_KNOBS, g_seed=G_SEED)
        k = (r["k1"], r["k2"], r["k3"])
        s_lin, sx_lin, sy_lin = envelope_linear(GEOM, k, sig0_x, sig0_y)
        s_sc, sx_sc, sy_sc = envelope_sc(dist, GEOM, k)
        # both devs from the SAME metric on the SAME sampling, so the two
        # columns are directly comparable; the solver's own roundness_dev uses a
        # coarser 21-point sweep and is not what is being compared here
        runs.append(dict(
            r=r, lin=(s_lin, sx_lin, sy_lin), sc=(s_sc, sx_sc, sy_sc),
            dev_lin=exit_roundness(s_lin, sx_lin, sy_lin, GEOM.s_q3_exit),
            dev_sc=exit_roundness(s_sc, sx_sc, sy_sc, GEOM.s_q3_exit)))
        print(f"  g1 = {g1:+5.2f} T/m done")

    print(f"\n{'g1':>6s} {'g2':>9s} {'g3':>9s} {'k1':>8s} {'k2':>8s} {'k3':>8s} "
          f"{'lin dev':>9s} {'sc dev':>9s} {'peak sx/mm':>11s} {'conv':>6s}")
    for run in runs:
        r, (_, sx_sc, _) = run["r"], run["sc"]
        print(f"{r['g1']:+6.2f} {r['g2']:+9.4f} {r['g3']:+9.4f} "
              f"{r['k1']:+8.3f} {r['k2']:+8.3f} {r['k3']:+8.3f} "
              f"{run['dev_lin']:9.2e} {run['dev_sc']:9.2e} "
              f"{sx_sc.max()*1e3:11.3f} {str(r['converged']):>6s}")
    worst = max(run["dev_sc"] for run in runs)
    print(f"\nlin dev / sc dev = worst fractional |sigma_x - sigma_y| anywhere in "
          f"the exit drift.\nThe linear solve drives its own metric to ~1e-4 by "
          f"construction; what matters is\nthat the SC column stays small at the "
          f"SAME knobs -- worst here {worst*100:.2f} %"
          f"{' (knobs were SC-solved too)' if SC_KNOBS else ''}.")

    # ------------------------------ plot ------------------------------
    apply_style()
    fig, (ax_l, ax_r) = plt.subplots(figsize=(6.4, 3.4), ncols=2, sharex=True,
                                     sharey=True, layout="constrained")
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(min(G1_SCAN), max(G1_SCAN))

    for ax, key, title in ((ax_l, "lin", "(a) space charge OFF (analytic)"),
                           (ax_r, "sc", "(b) space charge ON (tracked)")):
        # the exit drift, where the round condition is required to hold
        ax.axvspan(GEOM.s_q3_exit, GEOM.length, color="#c8c8c8", alpha=0.30,
                   lw=0, zorder=0)
        for s0, s1, _ in GEOM.element_spans():
            ax.axvspan(s0, s1, color="#8ecae6", alpha=0.45, lw=0, zorder=0)
        for run in runs:
            s, sx, sy = run[key]
            c = cmap(norm(run["r"]["g1"]))
            # linestyle is pinned: colour carries g1, dash the plane, whatever
            # the style's cycle does and would break solid=x / dashed=y
            ax.plot(s, sx * 1e3, color=c, lw=1.4, ls="-")
            ax.plot(s, sy * 1e3, color=c, lw=1.2, ls="--")
        ax.set_xlabel(r"$s$ [$m$]")
        ax.set_title(title, fontsize=10)
    ax_l.set_xlim(0.0, GEOM.length)
    ax_l.set_ylabel(r"$\sigma$ [$mm$]")

    ax_l.legend(handles=[Line2D([], [], color="0.3", lw=1.4, ls="-",
                                label=r"$\sigma_x$"),
                         Line2D([], [], color="0.3", lw=1.2, ls="--",
                                label=r"$\sigma_y$")],
                loc="upper left", fontsize=8, framealpha=0.85)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_r,
                 label=r"$g_1$ [$T/m$]")

    plt.show()
    save(fig, FIGS, 'fig_demo_triplet_round_transport')


if __name__ == "__main__":
    main()
