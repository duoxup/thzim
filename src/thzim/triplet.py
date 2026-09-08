#!/usr/bin/env python3
r"""Round-transport triplet: solve (g2, g3) so the beam leaves the triplet ROUND.

    marker  d0  Q1  d1  Q2  d2  Q3  d3 (exit)  marker

The building block of the two-triplet ("crossing") match in `thzim.two_triplet`.
Its job is NOT to hit a Twiss target -- that is what the match does -- but to
deliver a beam that is round, sigma_x = sigma_y, across the whole exit drift.
Roundness is what makes the middle region measurable with a single screen
reading per plane, and it is what lets two legs be joined by comparing two
numbers instead of four.

ONE public solver: `solve_round_triplet(dist, geom, g1)`.

## Why two conditions at the Q3 exit are enough

Impose, at the Q3 exit only,

    Sigma_x[0,0] = Sigma_y[0,0]      equal size,       set by g2
    Sigma_x[0,1] = Sigma_y[0,1]      equal divergence, set by g3

For equal x/y emittance those two give the third for free: det Sigma = eps^2
fixes Sigma[1,1] once Sigma[0,0] and Sigma[0,1] agree. All three second moments
then match, and since a drift transports them as

    sigma^2(s) = Sigma[0,0] + 2 s Sigma[0,1] + s^2 Sigma[1,1]

the beam is round at EVERY point of the exit drift, not just at the Q3 face.
Screens placed along that drift are verification markers, not constraints.

## Why the solve is staged, not a joint 2-knob fit

Three gradients against two conditions leave a one-parameter family. g1 is
taken as that parameter -- it is the knob the crossing match scans -- and
(g2, g3) are solved for each g1 by ALTERNATING two 1-D root finds:

    repeat:  g2 <- root of (sigma_x - sigma_y)          at the Q3 exit
             g3 <- root of (<xx'> - <yy'>)              at the Q3 exit

Each stage is monotonic and single-valued, so one fixed seed lands on the fully
round branch for every g1, with no branch jumps. The simultaneous two-knob
least-squares solve this replaced did jump branches partway through a g1 scan,
which is what made the scan curves unusable. The seed g3 = 0 means stage 1 runs
first, on an effectively two-quad configuration.

## Two fidelities

`sc=False` (default) is analytic 2x2 transport over `thzim.maps` -- no ocelot
import at all, microseconds per evaluation, which is what makes a fine g1 scan
practical.

`sc=True` repeats the same two stages with every moment taken from an ocelot
space-charge track, seeded from the linear answer so only three alternations
are needed. The same particles are re-used for every track, so each 1-D
objective stays deterministic and fsolve is not chasing statistical noise.

## Input contract: partdist only

Every function here that takes a bunch takes a **partdist**
`ParticleDistribution3D`, and nothing else. That is a deliberate narrowing --
the earlier version duck-typed ocelot `ParticleArray` as well, which put format
handling in every entry point and would have to grow a branch per new format.
Conversion belongs at the boundary instead:

    from partdist import from_ocelot_particle_array
    dist = from_ocelot_particle_array(pa)      # ocelot -> partdist, script side

The SC path still needs an ocelot array internally and converts one itself.

## This module is NOT energy-free

Unlike `thzim.dogleg`, whose whole design is geometric, the knobs here are
GRADIENTS [T/m] -- the machine setting -- converted to geometric k1 through the
beam rigidity. And the condition being solved is a property of the injected
beam's covariance, so a real distribution always enters. Both are deliberate:
this is the layer where the design meets an actual bunch.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, fsolve

from thzim.maps import drift_map, propagate, quad_map
from thzim.utils import M_E_GEV, brho

__all__ = [
    "SC_UNIT_STEP", "SC_MESH", "TripletGeom",
    "injection_sigma", "beam_energy_gev", "as_ocelot",
    "core_R", "roundness_dev", "solve_round_triplet",
]

# SC numerics defaults. Convergence-tested at 45 MeV / 4 nC: a 31^3 mesh and a
# 0.1 m step are resolution-robust to < 0.4 um and < 0.1 %. Pass a finer mesh or
# step where space charge is much stronger.
SC_UNIT_STEP = 0.1
SC_MESH = [31, 31, 31]

# The space-charge refinement searches only NEAR the linear answer -- half-width
# as a fraction of |g_linear|, floored so a near-zero gradient still gets a
# bracket, doubling until the residual changes sign and never past the max. See
# `_bounded_refine`.
SC_BRACKET_REL0 = 0.05
SC_BRACKET_RELMAX = 1.0
SC_BRACKET_FLOOR = 0.05          # [T/m]


# ------------------------------- the geometry -------------------------------

@dataclass
class TripletGeom:
    """Three-quad triplet with its lead-in and exit drifts (lengths in m).

    Layout `d0 Q1 d1 Q2 d2 Q3 d3`, so `drifts` is
    (before-Q1, Q1-Q2, Q2-Q3, exit). Only the first three enter the solve; the
    exit drift is where roundness is CHECKED, and the crossing match overrides
    it with the length of the middle region it has to cross.
    """

    lq: tuple = (0.15, 0.15, 0.15)             # Q1, Q2, Q3 magnetic lengths
    drifts: tuple = (0.30, 0.20, 0.20, 0.60)   # before-Q1, Q1-Q2, Q2-Q3, exit

    @property
    def L_exit(self):
        """Exit drift length [m] -- the default roundness-check span."""
        return self.drifts[3]

    @property
    def s_q3_exit(self):
        """Path length [m] from the entrance marker to the Q3 exit face."""
        return sum(self.lq) + sum(self.drifts[:3])

    @property
    def length(self):
        """Total length [m], entrance marker to exit marker."""
        return self.s_q3_exit + self.drifts[3]

    def element_spans(self):
        """(s_start, s_end, name) of each quadrupole, s from the entrance.

        Layout knowledge, so it lives with the geometry rather than in whichever
        script shades the magnets behind a curve.
        """
        spans, s = [], 0.0
        for i in range(3):
            s += self.drifts[i]
            spans.append((s, s + self.lq[i], f"Q{i + 1}"))
            s += self.lq[i]
        return spans

    def summary(self):
        return (
            f"Triplet  (Lq={self.lq} m, drifts={self.drifts} m)\n"
            f"  lengths: entrance->Q3 exit={self.s_q3_exit:.4f}  "
            f"exit drift={self.L_exit:.4f}  total={self.length:.4f} m")


def _bounded_refine(f, x0, rel0=SC_BRACKET_REL0, relmax=SC_BRACKET_RELMAX,
                    floor=SC_BRACKET_FLOOR, xtol=1e-4):
    """Root of `f` NEAR `x0`, never searched beyond a bounded neighbourhood.

    The space-charge stages refine a gradient that the linear solve has already
    placed, so the answer is a few per cent away and there is no reason to let a
    root find roam. Letting it roam is in fact fatal: an unbounded `fsolve` on an
    objective that TRACKS the beam will happily probe a gradient so large that a
    particle's transverse angle exceeds its own momentum, at which point ocelot's
    `xxstg_2_xp_mad` takes the square root of a negative number, the NaN reaches
    the space-charge mesh index, and `np.bincount` raises "'list' argument must
    have no negative elements" from deep inside `sc.py`. That is what this
    replaces.

    The bracket starts at `max(rel0 |x0|, floor)` and doubles until `f` changes
    sign or `max(relmax |x0|, floor)` is reached. A point where `f` cannot be
    evaluated (a track that failed or returned non-finite moments) is treated as
    outside the domain, not as a value.

    Returns (x, refined). `refined` is False when no sign change was found, and
    `x` is then `x0` unchanged -- a refinement that cannot be made is dropped,
    not forced.
    """
    f0 = f(x0)
    if not np.isfinite(f0):
        return x0, False
    if f0 == 0.0:
        return x0, True
    h = max(rel0 * abs(x0), floor)
    hmax = max(relmax * abs(x0), floor)
    while h <= hmax:
        for x in (x0 - h, x0 + h):
            fx = f(x)
            if np.isfinite(fx) and np.sign(fx) != np.sign(f0):
                lo, hi = (x, x0) if x < x0 else (x0, x)
                return float(brentq(f, lo, hi, xtol=xtol)), True
        h *= 2.0
    return x0, False


# ------------------------------ beam accessors ------------------------------
# partdist in, always. See "Input contract" in the module docstring.

def _require_partdist(dist):
    """Reject anything that is not a partdist distribution, helpfully."""
    if hasattr(dist, "rparticles"):
        raise TypeError(
            "thzim.triplet takes a partdist ParticleDistribution3D, not an "
            "ocelot ParticleArray. Convert at the call site:\n"
            "    from partdist import from_ocelot_particle_array\n"
            "    dist = from_ocelot_particle_array(pa)")
    if not all(hasattr(dist, a) for a in ("x", "xp", "y", "yp", "gamma0")):
        raise TypeError(
            f"expected a partdist ParticleDistribution3D, got "
            f"{type(dist).__name__} (missing x/xp/y/yp/gamma0)")


def injection_sigma(dist):
    """(Sigma_x, Sigma_y): the two 2x2 covariance blocks of `dist`.

    Units [m^2, m rad, rad^2]. `dist` is a partdist ParticleDistribution3D.
    """
    _require_partdist(dist)

    def block(u, up):
        u = np.asarray(u) - np.mean(u)
        up = np.asarray(up) - np.mean(up)
        return np.array([[np.mean(u * u), np.mean(u * up)],
                         [np.mean(u * up), np.mean(up * up)]])

    return block(dist.x, dist.xp), block(dist.y, dist.yp)


def beam_energy_gev(dist):
    """Total energy [GeV] of a partdist ParticleDistribution3D."""
    _require_partdist(dist)
    return dist.gamma0 * M_E_GEV


def as_ocelot(dist):
    """partdist distribution -> ocelot ParticleArray, referenced to its own z.

    `to_ocelot_particle_array` builds `tau = s - z` and stores `pa.s = s`, with
    `s` defaulting to 0. Left at the default, a bunch recorded at z = 7.4 m (the
    OP1 P0 beam) arrives with a mean `tau` of -7.4 m -- its 1.3 mm length sitting
    7.4 m from its own reference. The round trip through
    `from_ocelot_particle_array` is still exact, because `pa.s` records whatever
    was used, so nothing catches it; SPACE CHARGE does not survive it.

    `SpaceCharge.apply` works in MAD (constant-time) coordinates, and
    `xxstg_2_xp_mad` gets there by drifting every particle back by its own tau,
    `x_mad = x - x' beta tau`. A 7.4 m lever arm on sigma_x' = 2.1e-4 shears the
    bunch the field solver sees from sigma_x = 2.274 mm to 3.803 mm -- and then
    the inverse transform re-applies that lever arm to the KICKED momentum, so
    the position response to the space-charge kick comes back inverted and
    amplified: a +1e-6 rad outward kick moved x by -7.29 um instead of 0. In a
    plain 3 m drift from a waist that produced a bunch that grew and then shrank,
    which no repulsive force can do.

    So `s` is set here to the distribution's charge-weighted mean z: `tau` comes
    out centred AND `pa.s` still carries the true position, which keeps the round
    trip lossless. Verified against ocelot's own ASTRA adaptor -- 3 m drift from
    a 2.2785 mm waist, sigma_x 2.3900 mm here against 2.3901 mm there.

    The charge sign is partdist's job and it does it: `to_ocelot_particle_array`
    returns `abs(Q)`, because ocelot builds its density straight from `q_array`
    and a negative one makes the bunch attract itself. The check below is only a
    tripwire for an older partdist, since that failure is silent and looks
    plausible.
    """
    from partdist import to_ocelot_particle_array

    z = np.asarray(dist.z, dtype=float)
    w = np.abs(np.asarray(dist.get_data("Q"), dtype=float))
    zref = float(np.dot(w, z) / w.sum()) if w.sum() > 0 else float(z.mean())

    pa = to_ocelot_particle_array(dist, s=zref)
    if pa.q_array.sum() < 0:
        raise ValueError(
            "partdist returned a NEGATIVE q_array; ocelot's SpaceCharge builds "
            "its charge density straight from it, so the bunch would attract "
            "itself. Update partdist -- to_ocelot_particle_array must return "
            "abs(Q).")
    return pa


# ---------------------------- analytic transport ----------------------------

def core_R(geom, k, plane="x"):
    """2x2 map, triplet entrance -> Q3 EXIT, in one plane.

    `k` = (k1, k2, k3) geometric strengths [1/m^2]; `plane` is "x" or "y" (the
    same magnet focuses in one and defocuses in the other, so y just flips every
    sign). The exit drift is deliberately excluded -- see the module docstring
    for why the roundness conditions only need to hold at this face.
    """
    sign = 1.0 if plane == "x" else -1.0
    lq, d = geom.lq, geom.drifts
    R = drift_map(d[0])
    R = quad_map(sign * k[0], lq[0]) @ R
    R = drift_map(d[1]) @ R
    R = quad_map(sign * k[1], lq[1]) @ R
    R = drift_map(d[2]) @ R
    R = quad_map(sign * k[2], lq[2]) @ R
    return R


def roundness_dev(dist, geom, k, l_exit=None, nsamp=21):
    """Worst fractional |sigma_x - sigma_y| over the exit drift, linear optics.

    Sampled at `nsamp` points from the Q3 exit over `l_exit` (default
    `geom.L_exit`) and returned as max |sx - sy| / (0.5 (sx + sy)) -- so 0.02
    means "never more than 2 % out of round anywhere downstream", which is the
    figure `solve_round_triplet` reports and thresholds on.
    """
    s0x, s0y = injection_sigma(dist)
    Sx = propagate(core_R(geom, k, "x"), s0x)
    Sy = propagate(core_R(geom, k, "y"), s0y)
    s = np.linspace(0.0, geom.L_exit if l_exit is None else l_exit, nsamp)
    sx = np.sqrt(Sx[0, 0] + 2 * s * Sx[0, 1] + s**2 * Sx[1, 1])
    sy = np.sqrt(Sy[0, 0] + 2 * s * Sy[0, 1] + s**2 * Sy[1, 1])
    return float(np.max(np.abs(sx - sy) / (0.5 * (sx + sy))))


# -------------------------------- the solver --------------------------------

def solve_round_triplet(dist, geom, g1, sc=False, g_seed=(-2.0, 0.0), n_iter=8,
                        tol=0.02, unit_step=SC_UNIT_STEP, mesh=None):
    """Solve (g2, g3) for a ROUND beam out of the triplet, with g1 fixed.

    Two alternating 1-D root finds at the Q3 exit -- g2 for sigma_x = sigma_y,
    g3 for <xx'> = <yy'>. Decoupled and single-valued, so a fixed seed lands on
    the fully round branch for every g1 (no branch jumps). Knobs are gradients
    [T/m], converted to geometric k1 through the beam rigidity.

    sc=False (default)
        LINEAR analytic 2x2 transport -- fast, no ocelot.
    sc=True
        Refine UNDER SPACE CHARGE: the same two stages, but each moment comes
        from an ocelot SC track of `dist` at its own charge, seeded from the
        linear result so few tracks are needed. `dist` is converted to an ocelot
        ParticleArray here, once, and the same particles are re-used for every
        track. Each stage is a BOUNDED root find around the linear value
        (`_bounded_refine`); an unbounded one probes gradients that make the
        tracked beam non-physical and crashes inside ocelot's space-charge mesh.
        A stage that finds no sign change in its bracket leaves the gradient at
        the linear value rather than forcing one. The returned dict then also
        carries `g2_lin`, `g3_lin`.

    g_seed     initial (g2, g3) for the linear stages; g3 = 0 runs stage 1 first
    n_iter     linear stage alternations (the SC refine always uses 3)
    tol        `converged` is roundness_dev < tol
    unit_step  SC navigator step [m], sc=True only
    mesh       SC mesh, sc=True only (default SC_MESH)

    Returns dict(g1, g2, g3, k1, k2, k3, converged, roundness_dev
                 [, g2_lin, g3_lin]). `converged` is `roundness_dev < tol`; a g1
    with no round solution comes back with `converged=False` rather than
    raising or warning, since a scan is expected to walk into them.
    """
    sig_x, sig_y = injection_sigma(dist)
    energy_gev = beam_energy_gev(dist)
    rig = brho(energy_gev)
    lq, d = geom.lq, geom.drifts

    def lin_S_q3(g2, g3, plane, S0):
        return propagate(core_R(geom, (g1 / rig, g2 / rig, g3 / rig), plane), S0)

    g2, g3 = g_seed
    # A g1 with no round solution is an ORDINARY outcome here -- a scan walks
    # into them at the edges of the feasible window, and the caller is told
    # through `converged` / `roundness_dev`, which is the whole point of
    # reporting them. fsolve's "not making good progress" RuntimeWarning is
    # therefore duplicate information, and left unmuted it buries a scan's real
    # output. Only that class is silenced, and only around these two calls.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_iter):
            g2 = fsolve(lambda g: np.sqrt(lin_S_q3(g[0], g3, "x", sig_x)[0, 0])
                        - np.sqrt(lin_S_q3(g[0], g3, "y", sig_y)[0, 0]), [g2])[0]
            g3 = fsolve(lambda g: lin_S_q3(g2, g[0], "x", sig_x)[0, 1]
                        - lin_S_q3(g2, g[0], "y", sig_y)[0, 1], [g3])[0]
    extra = {}
    rdev = roundness_dev(dist, geom, (g1 / rig, g2 / rig, g3 / rig))

    if sc:
        import ocelot as oc
        pa0 = as_ocelot(dist)
        mesh = list(SC_MESH if mesh is None else mesh)

        def _track(g2, g3, exit_drift):
            seq = [oc.Marker(), oc.Drift(l=d[0]),
                   oc.Quadrupole(l=lq[0], k1=g1 / rig), oc.Drift(l=d[1]),
                   oc.Quadrupole(l=lq[1], k1=g2 / rig), oc.Drift(l=d[2]),
                   oc.Quadrupole(l=lq[2], k1=g3 / rig)]
            if exit_drift:
                seq.append(oc.Drift(l=d[3]))
            seq.append(oc.Marker())
            lat = oc.MagneticLattice(seq)
            navi = oc.Navigator(lat, unit_step=unit_step)
            navi.add_physics_proc(oc.SpaceCharge(1, nmesh_xyz=mesh),
                                  seq[0], seq[-1])
            return oc.track(lat, pa0.copy(), navi, print_progress=False)

        def sc_moments(g2, g3):
            """((<x^2>, <xx'>), (<y^2>, <yy'>)) at the Q3 exit, SC-tracked.

            Returns nans rather than raising when the track cannot be made
            sense of, so `_bounded_refine` reads that as "outside the domain"
            and stops expanding in that direction.
            """
            try:
                _, p = _track(g2, g3, exit_drift=False)
            except Exception:
                return (np.nan, np.nan), (np.nan, np.nan)
            rp = p.rparticles
            if not np.all(np.isfinite(rp)):
                return (np.nan, np.nan), (np.nan, np.nan)

            def mom(i):
                u = rp[i] - rp[i].mean()
                up = rp[i + 1] - rp[i + 1].mean()
                return np.mean(u * u), np.mean(u * up)

            return mom(0), mom(2)

        extra = dict(g2_lin=float(g2), g3_lin=float(g3))
        for _ in range(3):
            # stage 1: equal size. stage 2: equal divergence. Both searched only
            # near the linear answer -- see `_bounded_refine` for why.
            g2, _ = _bounded_refine(
                lambda g: (lambda a, b: np.sqrt(a[0]) - np.sqrt(b[0]))(
                    *sc_moments(g, g3)), g2)
            g3, _ = _bounded_refine(
                lambda g: (lambda a, b: a[1] - b[1])(
                    *sc_moments(g2, g)), g3)

        tws, _ = _track(g2, g3, exit_drift=True)   # SC roundness over the drift
        s_q3 = geom.s_q3_exit
        sx = np.array([np.sqrt(t.emit_x * t.beta_x) for t in tws
                       if t.s >= s_q3 - 1e-6])
        sy = np.array([np.sqrt(t.emit_y * t.beta_y) for t in tws
                       if t.s >= s_q3 - 1e-6])
        rdev = float(np.max(np.abs(sx - sy) / (0.5 * (sx + sy))))

    k1, k2, k3 = g1 / rig, g2 / rig, g3 / rig
    return dict(g1=g1, g2=float(g2), g3=float(g3),
                k1=k1, k2=float(k2), k3=float(k3),
                converged=bool(rdev < tol), roundness_dev=rdev, **extra)
