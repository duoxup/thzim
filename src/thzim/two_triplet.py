#!/usr/bin/env python3
r"""Two-triplet matching section: two round legs joined at a screen crossing.

    inj   T1(Q1 Q2 Q3)   GAP (screens)   T2(Q1 Q2 Q3)   target

The second of the two ways this package matches a beam onto a target Twiss; the
other is `thzim.quadruplet`. That one is the MODEL route -- four knobs against
four exit Twiss numbers, with the residual computed from the lattice. This one
is the OPERATIONAL route: what it matches on is the beam SIZE on the screens in
the middle, which is a quantity a real machine can actually read.

## The degree-of-freedom accounting is the whole idea

Six gradients, three per triplet. Four of them are spent on ROUNDNESS -- two per
leg, solved by `thzim.triplet.solve_round_triplet` -- leaving exactly two free:
the first quad of T1, and (running backwards from the target) the last quad of
T2.

Two knobs would normally be nowhere near enough for four Twiss constraints. But
because both legs arrive ROUND, sigma_x = sigma_y, agreeing on the beam size at
two positions in the middle drift means agreeing on the full covariance in BOTH
planes at once. The four constraints collapse to two, and the problem is exactly
determined. That is what buys the operational route its cheapness.

So: scan the two free gradients; each leg traces a curve in the
(sigma @ screen A, sigma @ screen C) plane; where the two curves CROSS, the
forward beam and the backward-propagated target beam are the same beam.

## Why a third screen is not decoration

Two sizes do NOT pin the beam by themselves. In a drift sigma^2(s) is a
parabola, `Sigma22 (s - s*)^2 + eps^2/Sigma22`, so with the emittance fixed it
still carries two free parameters -- and two different (waist size, waist
position) pairs can pass through the same two screen readings. The curves
therefore cross more than once, and only ONE of those crossings is the match.

Measured on `tools/demo_two_triplet_scan.py` (screens at 0.3 / 1.5 / 2.7 m
into a 3 m gap), reading the middle screen:

    crossing   sigma @ 1.5 m, fwd vs bwd     delivered Bmag
    0            0.2501 / 0.2473  (1.1 %)      1.016 / 1.005     <- the match
    1            1.1769 / 0.5831  (67 %)     302.4   / 224.6     <- spurious

The screens not used to form the crossing are what tell them apart, and it is
not a close call. `solve_crossing` scans every screen, forms the crossing from
`screen_pair`, and picks the candidate whose remaining screens agree
(`pick="consistent"`, the default). Taking the first crossing -- what the
upstream API did -- happens to be right here and is not right in general.

## Where space charge is on, and where it is not -- these are results, not defaults

  leg scans      SC ON, BOTH legs.  Not negotiable. It was tested: an
                 "anti-distortion" variant that ran the backward leg linearly
                 and reflected its SC displacement through the linear curve
                 (a retired upstream study, see MIGRATION.md) did NOT give a better
                 crossing. The screen sizes are what is being matched on, and
                 space charge changes them, so both legs must see it.
  round knobs    LINEAR by default, but CHECK -- and the check is free. g2 and
                 g3 enforce a RATIO between the planes, and the SC kick of an
                 already-round beam is symmetric, so it barely moves that ratio:
                 1.6 % worst deviation at 39.5 MeV through 2.1 m
                 (`tools/demo_triplet_round_transport.py`). On the present
                 two-triplet geometry the chosen OP1 crossing is within 4.5 %.
                 Re-solving under space charge did not pay for itself upstream
                 either (another retired study). Still, `leg_scan` reports
                 `sc_roundness_dev` per scan point, read straight off the track
                 it already ran, and `sc_knobs=True` re-solves g2 and g3 under
                 space charge (seeded from the linear answer) if that number
                 ever says it is needed, at a couple of minutes per point. Do look
                 at it: a first run showed 13-27 %, which turned out to be a
                 broken partdist -> ocelot conversion rather than physics (see
                 `thzim.triplet.as_ocelot`), and the number is what made it
                 visible.
  end-to-end     SC ON. That is the validation, so it must be honest.

## Physical order only

`TwoTripletGeom` is specified, and reports, in PHYSICAL order: T1 is Q1, Q2, Q3
downstream, and so is T2. The backward leg needs the reversed T2 -- and its
"first" quad is then T2's physical Q3 -- but that reversal happens inside this
module and never reaches the caller. Gradients go in and come out as
`(Q1, Q2, Q3)` per triplet.

## Status

This route SUPERSEDES the earlier six-knob joint least squares over two quad
groups (retired upstream, not in this package) and is the adopted OP1/OP2
upstream-matching route. See MIGRATION.md for the design-change history.

Beams are partdist distributions; ocelot conversion happens internally.
"""

from dataclasses import dataclass

import numpy as np
import ocelot as oc
from ocelot.cpbd.physics_proc import PhysProc

from thzim.maps import propagate
from thzim.triplet import (TripletGeom, as_ocelot, beam_energy_gev,
                           core_R, injection_sigma, roundness_dev,
                           solve_round_triplet)
from thzim.utils import brho

__all__ = [
    "SC_UNIT_STEP", "SC_MESH", "TwoTripletGeom", "TwoTripletMatch",
    "backward_beam", "leg_scan", "find_crossing", "find_crossings",
    "match_at", "match_with", "solve_crossing",
    "build_lattice", "track_e2e", "bmag",
]

SC_UNIT_STEP = 0.05
SC_MESH = (31, 31, 31)


class _Sampler(PhysProc):
    """A no-op physics process, attached only to make the navigator step.

    ocelot subdivides the lattice at `unit_step` ONLY where a physics process is
    active. With space charge off there is none, so the twiss output lands on
    element boundaries alone -- about a dozen points across a matching section,
    which draws as a polygon no matter what `unit_step` is set to. `apply` does
    nothing, and splitting a linear map in two is exact, so the tracked beam is
    untouched; this only changes how often it is sampled.
    """

    def apply(self, p, dz):
        pass



# ------------------------------- the geometry -------------------------------

@dataclass
class TwoTripletGeom:
    """Two triplets around a screen section, in PHYSICAL order throughout.

    `t1_drifts` is (lead-in, Q1-Q2, Q2-Q3); `t2_drifts` is
    (Q1-Q2, Q2-Q3, lead-out) -- each triplet's three drifts in the order the
    beam meets them. `gap` is T1's Q3 exit to T2's Q1 entrance, and `screens`
    are offsets into that gap measured from the T1 Q3 exit.
    """

    t1_lq: tuple = (0.15, 0.15, 0.15)
    t1_drifts: tuple = (0.50, 0.15, 0.15)
    t2_lq: tuple = (0.15, 0.15, 0.15)
    t2_drifts: tuple = (0.15, 0.15, 0.20)
    gap: float = 1.0
    screens: tuple = (0.10, 0.50, 0.90)

    # -- the two legs, as the round solver sees them ------------------------
    @property
    def t1_leg(self):
        """T1 as a `TripletGeom`, exit drift = the gap. Same order as physical."""
        return TripletGeom(lq=tuple(self.t1_lq),
                           drifts=(self.t1_drifts[0], self.t1_drifts[1],
                                   self.t1_drifts[2], self.gap))

    @property
    def t2_leg(self):
        """T2 REVERSED, exit drift = the gap: the leg run back from the target.

        Everything is flipped -- the lead-out drift becomes the lead-in, and the
        leg's first quad is T2's physical Q3. Callers never see this ordering;
        `solve_crossing` converts back before returning.
        """
        return TripletGeom(lq=tuple(self.t2_lq[::-1]),
                           drifts=(self.t2_drifts[2], self.t2_drifts[1],
                                   self.t2_drifts[0], self.gap))

    # -- positions along the line -------------------------------------------
    @property
    def s_gap_start(self):
        """s of the T1 Q3 exit [m] -- where the screen offsets are measured from."""
        return sum(self.t1_lq) + sum(self.t1_drifts)

    @property
    def s_gap_end(self):
        """s of the T2 Q1 entrance [m]."""
        return self.s_gap_start + self.gap

    @property
    def length(self):
        """Total length [m], injection plane to target plane."""
        return self.s_gap_end + sum(self.t2_lq) + sum(self.t2_drifts)

    @property
    def screen_s(self):
        """Absolute s [m] of each screen."""
        return tuple(self.s_gap_start + p for p in self.screens)

    @property
    def bwd_screen_offsets(self):
        """Screen offsets seen from the BACKWARD leg's own Q3 exit [m].

        Same screens, same labels, measured from the other end of the gap.
        """
        return tuple(self.gap - p for p in self.screens)

    def element_spans(self):
        """(s_start, s_end, name) of every quad, physical order, s from the start."""
        spans, s = [], 0.0
        for i in range(3):
            s += self.t1_drifts[i]
            spans.append((s, s + self.t1_lq[i], f"T1.Q{i + 1}"))
            s += self.t1_lq[i]
        s += self.gap
        for i in range(3):
            spans.append((s, s + self.t2_lq[i], f"T2.Q{i + 1}"))
            s += self.t2_lq[i] + self.t2_drifts[i]
        return spans

    def summary(self):
        return (
            f"Two-triplet section  (gap={self.gap:.3f} m, "
            f"screens at {tuple(round(p, 3) for p in self.screens)} m into it)\n"
            f"  T1  Lq={self.t1_lq} m  drifts (lead-in, Q1-Q2, Q2-Q3)"
            f"={self.t1_drifts} m\n"
            f"  T2  Lq={self.t2_lq} m  drifts (Q1-Q2, Q2-Q3, lead-out)"
            f"={self.t2_drifts} m\n"
            f"  gap {self.s_gap_start:.4f} -> {self.s_gap_end:.4f} m, "
            f"total length {self.length:.4f} m")


def bmag(beta, alpha, beta0, alpha0=0.0):
    """Twiss mismatch amplification against (beta0, alpha0); 1.0 is matched."""
    g0 = (1.0 + alpha0**2) / beta0
    g = (1.0 + alpha**2) / beta
    return 0.5 * (beta0 * g - 2.0 * alpha0 * alpha + g0 * beta)


# --------------------------------- the beams ---------------------------------

def backward_beam(dist, target):
    """`dist` re-conditioned to sit at the target plane, running backwards.

    `target` is the forward-sense (beta_x, alpha_x, beta_y, alpha_y) wanted at
    the end of the section; the alphas are NEGATED here, which is what time
    reversal does to them.

    Built from the injection distribution rather than from a synthetic Gaussian,
    so the two legs share an emittance and a longitudinal profile exactly --
    `partdist.match_twiss_xy` is symplectic per plane, so it changes the
    transverse optics and nothing else.
    """
    from partdist.pd3d.manipulator import match_twiss_xy

    beta_x, alpha_x, beta_y, alpha_y = target
    return match_twiss_xy(dist, alpha_x=-alpha_x, beta_x=beta_x,
                          alpha_y=-alpha_y, beta_y=beta_y)


# ------------------------------- the leg scan -------------------------------

def _leg_lattice(leg, k, energy_gev):
    """Triplet + gap as an ocelot lattice, in the leg's own (model) order."""
    lq, d = leg.lq, leg.drifts
    return oc.MagneticLattice([
        oc.Marker(), oc.Drift(l=d[0]),
        oc.Quadrupole(l=lq[0], k1=k[0]), oc.Drift(l=d[1]),
        oc.Quadrupole(l=lq[1], k1=k[1]), oc.Drift(l=d[2]),
        oc.Quadrupole(l=lq[2], k1=k[2]), oc.Drift(l=d[3]), oc.Marker()])


def _envelope_dev(tws, s_from):
    """Worst fractional |sigma_x - sigma_y| from `s_from` onward, off a track."""
    s = np.array([t.s for t in tws])
    sx = np.sqrt(np.array([t.emit_x * t.beta_x for t in tws]))
    sy = np.sqrt(np.array([t.emit_y * t.beta_y for t in tws]))
    m = s >= s_from - 1e-9
    return float(np.max(np.abs(sx[m] - sy[m]) / (0.5 * (sx[m] + sy[m]))))


def _screen_sigma_sc(dist, leg, k, offsets, unit_step, mesh):
    """(mean sigma [m] at each screen, the ocelot twiss list) from an SC track."""
    lat = _leg_lattice(leg, k, beam_energy_gev(dist))
    navi = oc.Navigator(lat, unit_step=unit_step)
    navi.add_physics_proc(oc.SpaceCharge(1, nmesh_xyz=list(mesh)),
                          lat.sequence[0], lat.sequence[-1])
    tws, _ = oc.track(lat, as_ocelot(dist), navi, print_progress=False)
    s = np.array([t.s for t in tws])
    sx = np.sqrt(np.array([t.emit_x * t.beta_x for t in tws]))
    sy = np.sqrt(np.array([t.emit_y * t.beta_y for t in tws]))
    at = leg.s_q3_exit + np.asarray(offsets)
    return 0.5 * (np.interp(at, s, sx) + np.interp(at, s, sy)), tws


def _screen_sigma_linear(dist, leg, k, offsets):
    """Mean sigma [m] at each screen, analytic -- exact, no sampling.

    The screens sit in a drift, so the covariance at the Q3 exit gives them in
    closed form; nothing has to be stepped along.
    """
    s0x, s0y = injection_sigma(dist)
    out = []
    for S0, plane in ((s0x, "x"), (s0y, "y")):
        S = propagate(core_R(leg, k, plane), S0)
        d = np.asarray(offsets)
        out.append(np.sqrt(S[0, 0] + 2 * d * S[0, 1] + d**2 * S[1, 1]))
    return 0.5 * (out[0] + out[1])


def _screen_row(dist, leg, k, offsets, sc, unit_step, mesh):
    """(sigma at each offset, tws, SC roundness dev) for GIVEN strengths `k`.

    The half of `leg_scan` that does not solve anything, so `match_with` can
    evaluate a lattice it was handed rather than re-deriving one.
    """
    if sc:
        sigma, tws = _screen_sigma_sc(dist, leg, k, offsets, unit_step, mesh)
        return np.asarray(sigma), tws, _envelope_dev(tws, leg.s_q3_exit)
    return np.asarray(_screen_sigma_linear(dist, leg, k, offsets)), None, np.nan


def leg_scan(dist, leg, g1_values, offsets, sc=True, sc_knobs=False,
             round_tol=0.05, unit_step=SC_UNIT_STEP, mesh=SC_MESH,
             g_seed=(-2.0, 0.0), warm_start=True, keep_tws=False,
             verbose=False):
    """Scan one leg's free gradient; return the knobs and the screen sizes.

    Per `g1`: solve (g2, g3) for a round exit beam, then read the mean sigma at
    each `offsets` position beyond the Q3 exit, tracked with space charge
    (`sc=True`, the default and the only setting the crossing should be taken
    from) or analytically (`sc=False`, for comparison and for cheap dry runs).

    The round solve is LINEAR unless `sc_knobs=True`, which re-solves it under
    space charge (seeded from the linear answer, so it converges in a few
    tracks) at a couple of minutes per point. Which one is right is a property
    of
    the section, not a preference -- see the module docstring, and read
    `sc_roundness_dev` below before deciding.

    With `sc=True` each row also carries `sc_roundness_dev`: the worst
    fractional |sigma_x - sigma_y| anywhere beyond the Q3 exit, measured on the
    track that was run anyway. It costs nothing and it is the number that says
    whether the linear knobs are good enough here. If it is a few per cent the
    crossing is on solid ground; if it is tens of per cent the beam is not round
    in the middle at all, and the two-screens-pin-four-Twiss argument this whole
    module rests on does not apply until `sc_knobs` is turned on.

    The screen reading is the MEAN of sigma_x and sigma_y. The beam is round
    there, so the two are the same quantity measured twice, and averaging halves
    the statistical noise the crossing has to be found through.

    A round solution does NOT exist for every g1 -- push it far enough and the
    staged root find stops converging. Those points get `sigma = nan` rather
    than a fabricated size, so `find_crossing` skips the segments touching them
    instead of crossing a curve that means nothing.

    But most apparent failures are the SEED, not the physics.
    `solve_round_triplet`'s claim that one fixed seed always lands on the round
    branch holds on the geometry it was written for and not in general: on the
    OP1 section (weaker quads, a 3.4 m middle) the default seed misses solutions
    that plainly exist --

        g1 [T/m]     -0.479   -0.414   -0.350   -0.286   -0.221
        seed (-2,0)   1.5e-2   2.3e-1   3.6e-1   5.0e-1   1.2e-4
        seed (+1,0)   1.5e-2   2.4e-2   1.9e-3   2.8e-4   1.2e-4

    -- and the hole it punched swallowed the correct crossing. So each g1 is
    tried from up to three seeds: the previous g1's answer (`warm_start`, which
    follows the branch continuously, the natural thing for a scan), the given
    `g_seed`, and `g_seed` with its first component flipped. The roundest result
    wins. This costs two extra 2x2 solves per point, which is nothing, and it
    does not change any answer the fixed seed already found.

    `round_tol` is the DROP threshold, and it is deliberately looser than
    `solve_round_triplet`'s own 2 % `converged` flag. The screen reading is the
    mean of sigma_x and sigma_y, and the crossing argument -- round beam, so two
    sizes pin four Twiss numbers -- degrades smoothly with the roundness: at 2 %
    out of round it is accurate to 2 %, which is fine, while at 50 % it means
    nothing. Dropping at the solver's 2 % flag instead punches holes in the
    curve at points that were merely marginal, and a hole can swallow the
    crossing segment (observed). Rows keep the solver's `converged` flag as a
    quality marker either way.

    `keep_tws=True` (needs `sc=True`) also stores ocelot's twiss list for each
    tracked leg, which the scan already has in hand -- so a script that wants to
    draw the envelopes does not have to re-track for them, and derives whatever
    it needs from ocelot's own output.

    Returns a list of dicts: g1, g2, g3, k (the three k1), sigma (one per
    offset), roundness_dev, converged, sc_roundness_dev (nan unless `sc`), and
    tws when `keep_tws`.
    """
    rows, warm = [], None
    for g1 in g1_values:
        seeds = [g_seed, (-g_seed[0], g_seed[1])]
        if warm_start and warm is not None:
            seeds.insert(0, warm)
        r = None
        for seed in seeds:
            trial = solve_round_triplet(dist, leg, float(g1), sc=sc_knobs,
                                        g_seed=seed, unit_step=unit_step,
                                        mesh=list(mesh))
            if r is None or trial["roundness_dev"] < r["roundness_dev"]:
                r = trial
            if r["roundness_dev"] < round_tol:
                break                      # good enough; do not keep searching
        if r["roundness_dev"] < round_tol:
            warm = (r["g2"], r["g3"])
        k = (r["k1"], r["k2"], r["k3"])
        if not (r["roundness_dev"] < round_tol):     # NaN-safe
            sigma, tws, sc_dev = np.full(len(offsets), np.nan), None, np.nan
        else:
            sigma, tws, sc_dev = _screen_row(dist, leg, k, offsets, sc,
                                             unit_step, mesh)
        row = dict(g1=float(g1), g2=r["g2"], g3=r["g3"], k=k, sigma=sigma,
                   roundness_dev=r["roundness_dev"], converged=r["converged"],
                   sc_roundness_dev=sc_dev)
        if keep_tws:
            row["tws"] = tws
        rows.append(row)
        if verbose:
            tag = ("  ".join(f"{v*1e3:.4f}" for v in sigma) + " mm"
                   + ("" if r["converged"] else "   (marginal)")
                   + (f"   SC round dev {sc_dev*100:5.1f} %"
                      if np.isfinite(sc_dev) else "")
                   if np.isfinite(sigma[0])
                   else f"NOT ROUND (dev {r['roundness_dev']:.2e}) -- dropped")
            print(f"    g1 = {g1:+7.3f} T/m -> sigma {tag}")
    n_bad = sum(1 for r in rows if not np.isfinite(r["sigma"][0]))
    if n_bad and verbose:
        print(f"    ({n_bad}/{len(rows)} g1 values had no round solution "
              f"within {round_tol:.0%})")
    devs = [r["sc_roundness_dev"] for r in rows
            if np.isfinite(r["sc_roundness_dev"])]
    if devs and verbose:
        worst = max(devs)
        note = ("" if worst < 0.05 else
                "  <- the beam is NOT round in the middle; the crossing "
                "argument does not\n       hold here. Re-run with "
                "sc_knobs=True.")
        print(f"    worst SC roundness deviation over the scan: "
              f"{worst*100:.1f} %{note}")
    return rows


def find_crossings(fwd_xy, g1_fwd, bwd_xy, g1_bwd):
    """EVERY intersection of the two polylines in the screen-size plane.

    `fwd_xy` / `bwd_xy` are (n, 2) arrays of (sigma @ screen A, sigma @ screen C)
    along each scan; `nan` rows (no round solution there) break the polyline
    rather than being interpolated across. Returns a list of
    (g1_fwd, g1_bwd, sigma_A, sigma_C), ordered along the forward scan.

    There is usually MORE THAN ONE, and the extras are SPURIOUS, not alternative
    matches -- two screen readings do not pin a beam on their own. See the
    module docstring. This function is pure geometry and does not judge them;
    `solve_crossing` scores each against the screens it did not use.
    """
    F, B = np.asarray(fwd_xy, float), np.asarray(bwd_xy, float)
    out = []
    for i in range(len(F) - 1):
        p1, p2 = F[i], F[i + 1]
        if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
            continue
        r = p2 - p1
        for j in range(len(B) - 1):
            q1, q2 = B[j], B[j + 1]
            if not (np.all(np.isfinite(q1)) and np.all(np.isfinite(q2))):
                continue
            s = q2 - q1
            rxs = r[0] * s[1] - r[1] * s[0]
            if abs(rxs) < 1e-15:                 # parallel segments
                continue
            qp = q1 - p1
            t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
            u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                gf = g1_fwd[i] + t * (g1_fwd[i + 1] - g1_fwd[i])
                gb = g1_bwd[j] + u * (g1_bwd[j + 1] - g1_bwd[j])
                xc, yc = p1 + t * r
                out.append((float(gf), float(gb), float(xc), float(yc)))
    return out


def find_crossing(fwd_xy, g1_fwd, bwd_xy, g1_bwd):
    """The first intersection, or None. See `find_crossings` for the rest."""
    all_of_them = find_crossings(fwd_xy, g1_fwd, bwd_xy, g1_bwd)
    return all_of_them[0] if all_of_them else None


# ------------------------------- the solution -------------------------------

@dataclass
class TwoTripletMatch:
    """Result of a two-triplet crossing match. Gradients are in PHYSICAL order."""

    geom: TwoTripletGeom
    t1_g: tuple                    # T1 (Q1, Q2, Q3) gradients [T/m]
    t2_g: tuple                    # T2 (Q1, Q2, Q3) gradients [T/m]
    energy_gev: float
    target: tuple                  # forward-sense Twiss wanted at the end
    screen_sigma: tuple            # (sigma @ A, sigma @ C) at the crossing [m]
    screen_pair: tuple             # which two screens the crossing used
    sc_scan: bool
    crossings: list = None         # every intersection: dicts with g1_fwd,
                                   # g1_bwd, sigma, residual (see solve_crossing)
    picked: int = 0                # which of them this match is
    screen_residual: float = None  # worst fwd/bwd disagreement at the screens
                                   # NOT used to form the crossing; nan if none
    fwd_sigma: tuple = None        # each leg's sigma at EVERY screen [m], so the
    bwd_sigma: tuple = None        # per-screen agreement can be re-examined
    lin_round_dev: float = None    # worst LINEAR roundness deviation of either
                                   # leg -- for match_with, the check that the
                                   # gradients handed in really are round
    round_dev: float = None        # worst SC roundness deviation of either leg
                                   # beyond its Q3 exit; nan if not SC-tracked
    sc_knobs: bool = False         # were g2, g3 solved under space charge
    fwd_rows: list = None          # the raw leg scans, for plotting
    bwd_rows: list = None
    exit_twiss: tuple = None       # filled in by track_e2e

    @property
    def t1_k1s(self):
        b = brho(self.energy_gev)
        return [g / b for g in self.t1_g]

    @property
    def t2_k1s(self):
        b = brho(self.energy_gev)
        return [g / b for g in self.t2_g]

    def bmag(self):
        """(Bmag_x, Bmag_y) of the delivered Twiss; needs `track_e2e` first."""
        if self.exit_twiss is None:
            return None
        bx, ax, by, ay = self.exit_twiss
        return (bmag(bx, ax, self.target[0], self.target[1]),
                bmag(by, ay, self.target[2], self.target[3]))

    def summary(self):
        lines = [
            f"Two-triplet match ({'SC' if self.sc_scan else 'linear'} leg scans,"
            f" {'SC' if self.sc_knobs else 'linear'} round knobs, "
            f"E={self.energy_gev*1e3:.1f} MeV)",
            f"  target   bx={self.target[0]:.3f} ax={self.target[1]:+.3f}  "
            f"by={self.target[2]:.3f} ay={self.target[3]:+.3f}",
            "  crossing sigma = ("
            + ", ".join(f"{v*1e3:.4f}" for v in self.screen_sigma)
            + f") mm at screens {self.screen_pair}"
            + (f"   [#{self.picked} of {len(self.crossings)}]"
               if self.crossings and len(self.crossings) > 1 else "")]
        if (self.lin_round_dev is not None
                and np.isfinite(self.lin_round_dev)
                and self.lin_round_dev >= 0.05):
            lines.append(
                f"  !! these gradients are NOT a round solution: linear "
                f"roundness deviation {self.lin_round_dev*100:.1f} %")
        if self.round_dev is not None and np.isfinite(self.round_dev):
            lines.append(
                f"  roundness under SC: worst |sigma_x - sigma_y| beyond a Q3 "
                f"exit is {self.round_dev*100:.1f} %"
                + ("" if self.round_dev < 0.05 else
                   "  <- the beam is NOT round in the\n"
                   "      middle, so two screens do not pin it; re-solve with "
                   "sc_knobs=True"))
        if self.screen_residual is not None and np.isfinite(self.screen_residual):
            lines.append(f"  free-screen check: fwd and bwd agree to "
                         f"{self.screen_residual*100:.2f} % (the crossing is "
                         f"{'consistent' if self.screen_residual < 0.05 else 'SUSPECT'})")
        for tag, gs, ks in (("T1", self.t1_g, self.t1_k1s),
                            ("T2", self.t2_g, self.t2_k1s)):
            for i, (g, k) in enumerate(zip(gs, ks)):
                lines.append(f"  {tag}.Q{i+1}  g={g:+8.4f} T/m   "
                             f"k1={k:+9.3f} 1/m^2")
        if self.exit_twiss is not None:
            bmx, bmy = self.bmag()
            lines.append(
                f"  delivered bx={self.exit_twiss[0]:.3f} "
                f"ax={self.exit_twiss[1]:+.3f}  by={self.exit_twiss[2]:.3f} "
                f"ay={self.exit_twiss[3]:+.3f}   Bmag={bmx:.3f}/{bmy:.3f}")
        return "\n".join(lines)


def match_at(dist, geom, target, g1_fwd, g1_bwd, sc=True, sc_knobs=False,
             screen_pair=(0, -1), round_tol=0.05,
             unit_step=SC_UNIT_STEP, mesh=SC_MESH,
             g_seed=(-2.0, 0.0), g_seed_bwd=None, bwd=None):
    """Build the match at a CHOSEN pair of free gradients -- no scan, no search.

    Solves each leg's round knobs at the given g1, reads every screen, and
    reports how well the two legs agree there. This is the entry point when the
    working point has already been picked off a scan (`leg_scan` +
    `find_crossings`), which is how the demo scripts are split: one presents the
    scan, a person chooses, the other runs the choice.

    `screen_residual` is the worst relative disagreement over the screens NOT in
    `screen_pair` -- the independent check that the two legs really describe one
    beam, since two readings alone cannot pin it (module docstring). It is `nan`
    if either leg found no round solution here, which means the working point is
    unusable, not merely unchecked.

    `g_seed` / `g_seed_bwd` seed the round solve per leg (see `leg_scan` on why
    the seed matters); `bwd` reuses a backward beam you already built.

    Returns a `TwoTripletMatch` with `crossings=None`; run `track_e2e` on it for
    the delivered Twiss and Bmag.
    """
    if bwd is None:
        bwd = backward_beam(dist, target)
    n_scr = len(geom.screens)
    i, j = (screen_pair[0] % n_scr, screen_pair[1] % n_scr)
    kw = dict(sc=sc, sc_knobs=sc_knobs, round_tol=round_tol,
              unit_step=unit_step, mesh=mesh)
    rf = leg_scan(dist, geom.t1_leg, [g1_fwd], tuple(geom.screens),
                  g_seed=g_seed, **kw)[0]
    rb = leg_scan(bwd, geom.t2_leg, [g1_bwd], tuple(geom.bwd_screen_offsets),
                  g_seed=g_seed if g_seed_bwd is None else g_seed_bwd, **kw)[0]

    worst = np.nan
    for c in (c for c in range(n_scr) if c not in (i, j)):
        sf, sb = rf["sigma"][c], rb["sigma"][c]
        d = abs(sf - sb) / (0.5 * (sf + sb))
        worst = d if not np.isfinite(worst) else max(worst, d)
    return TwoTripletMatch(
        geom=geom,
        t1_g=(rf["g1"], rf["g2"], rf["g3"]),        # leg order == physical
        t2_g=(rb["g3"], rb["g2"], rb["g1"]),        # leg order reversed
        energy_gev=beam_energy_gev(dist), target=tuple(target),
        screen_sigma=tuple(0.5 * (rf["sigma"][c] + rb["sigma"][c])
                           for c in (i, j)),
        screen_pair=(i, j), fwd_sigma=tuple(rf["sigma"]),
        bwd_sigma=tuple(rb["sigma"]), sc_scan=bool(sc),
        screen_residual=worst,
        lin_round_dev=max(rf["roundness_dev"], rb["roundness_dev"]),
        round_dev=float(np.nanmax([rf["sc_roundness_dev"],
                                   rb["sc_roundness_dev"]]))
        if np.isfinite(rf["sc_roundness_dev"]) else np.nan,
        sc_knobs=bool(sc_knobs))


def match_with(dist, geom, target, t1_g, t2_g, sc=True, screen_pair=(0, -1),
               unit_step=SC_UNIT_STEP, mesh=SC_MESH, bwd=None):
    """Evaluate a GIVEN pair of gradient triples -- no solve, no search.

    `t1_g` / `t2_g` are (Q1, Q2, Q3) gradients [T/m] in PHYSICAL order, exactly
    as `TwoTripletMatch` reports them and as the scan script tabulates them.

    Use this, not `match_at`, to reproduce a working point picked off a scan.
    **A g1 does not determine the lattice.** The round solve at a fixed g1 has
    several branches and which one it lands on depends on the seed: a scan finds
    the right one by continuity from its neighbours, and `match_at` called cold
    at the same g1 has nothing to be continuous with. On the OP1 section, g1_bwd
    = +0.3542 gives T2 = (-0.579, +0.582, +0.354) T/m from the scan and
    (+0.357, +5.648, +0.354) cold -- a 106 1/m^2 quadrupole, a free-screen
    residual of 180 % against 0.00 %, and an emittance blow-up downstream.
    Passing the gradients removes the ambiguity entirely.

    `lin_round_dev` on the result is the check that what was handed in really is
    a round solution; `screen_residual` and `round_dev` mean what they always do.
    """
    if bwd is None:
        bwd = backward_beam(dist, target)
    energy_gev = beam_energy_gev(dist)
    b_rho = brho(energy_gev)
    n_scr = len(geom.screens)
    i, j = (screen_pair[0] % n_scr, screen_pair[1] % n_scr)

    kf = tuple(g / b_rho for g in t1_g)                 # T1: leg == physical
    kb = tuple(g / b_rho for g in reversed(t2_g))       # T2: leg is reversed
    sf, tf, devf = _screen_row(dist, geom.t1_leg, kf, tuple(geom.screens),
                               sc, unit_step, mesh)
    sb, tb, devb = _screen_row(bwd, geom.t2_leg, kb,
                               tuple(geom.bwd_screen_offsets), sc, unit_step,
                               mesh)
    worst = np.nan
    for c in (c for c in range(n_scr) if c not in (i, j)):
        d = abs(sf[c] - sb[c]) / (0.5 * (sf[c] + sb[c]))
        worst = d if not np.isfinite(worst) else max(worst, d)
    return TwoTripletMatch(
        geom=geom, t1_g=tuple(t1_g), t2_g=tuple(t2_g), energy_gev=energy_gev,
        target=tuple(target),
        screen_sigma=tuple(0.5 * (sf[c] + sb[c]) for c in (i, j)),
        screen_pair=(i, j), sc_scan=bool(sc), screen_residual=worst,
        fwd_sigma=tuple(sf), bwd_sigma=tuple(sb),
        lin_round_dev=max(roundness_dev(dist, geom.t1_leg, kf),
                          roundness_dev(bwd, geom.t2_leg, kb)),
        round_dev=float(np.nanmax([devf, devb])) if np.isfinite(devf) else np.nan)


def solve_crossing(dist, geom, target, g1_fwd_scan, g1_bwd_scan, sc=True,
                   sc_knobs=False, screen_pair=(0, -1), round_tol=0.05,
                   unit_step=SC_UNIT_STEP, mesh=SC_MESH,
                   g_seed=(-2.0, 0.0), pick="consistent", keep_tws=False,
                   verbose=True):
    """Match `dist` onto `target` by crossing the two legs' screen curves.

    Scans both free gradients, finds where the (sigma @ A, sigma @ C) curves
    meet, and re-solves the round knobs there. `screen_pair` picks which two of
    `geom.screens` span the crossing plane -- the outer two by default, since a
    longer lever arm separates the curves better.

    EVERY screen is scanned, but only the two named by `screen_pair` form the
    crossing -- the rest are the check that tells a real crossing from a
    spurious one (see the module docstring; the difference is 1 % against 67 %
    on the demo case, and Bmag 1.02 against 302). `pick="consistent"` (the
    default) takes the candidate whose free screens agree best; an integer picks
    by index instead. `.crossings` carries them all with their residuals, so the
    decision stays visible.

    Returns a `TwoTripletMatch` with PHYSICAL-order gradients, or None when the
    curves do not cross in the given windows (widen or move them). Run
    `track_e2e` on the result to fill in the delivered Twiss and Bmag.
    """
    bwd = backward_beam(dist, target)
    n_scr = len(geom.screens)
    i, j = (screen_pair[0] % n_scr, screen_pair[1] % n_scr)
    fwd_off = tuple(geom.screens)
    bwd_off = tuple(geom.bwd_screen_offsets)
    free = [c for c in range(n_scr) if c not in (i, j)]

    if verbose:
        print(geom.summary())
        print(f"  scanning forward leg ({len(g1_fwd_scan)} points, "
              f"SC {'on' if sc else 'off'}) ...")
    fwd = leg_scan(dist, geom.t1_leg, g1_fwd_scan, fwd_off, sc=sc,
                   sc_knobs=sc_knobs, round_tol=round_tol,
                   unit_step=unit_step, mesh=mesh, g_seed=g_seed,
                   keep_tws=keep_tws, verbose=verbose)
    if verbose:
        print(f"  scanning backward leg ({len(g1_bwd_scan)} points) ...")
    bwd_rows = leg_scan(bwd, geom.t2_leg, g1_bwd_scan, bwd_off, sc=sc,
                        sc_knobs=sc_knobs, round_tol=round_tol,
                        unit_step=unit_step, mesh=mesh, g_seed=g_seed,
                        keep_tws=keep_tws, verbose=verbose)

    gf_arr, gb_arr = np.asarray(g1_fwd_scan), np.asarray(g1_bwd_scan)
    F = np.array([r["sigma"] for r in fwd])
    B = np.array([r["sigma"] for r in bwd_rows])
    raw = find_crossings(F[:, [i, j]], gf_arr, B[:, [i, j]], gb_arr)
    if not raw:
        if verbose:
            print("  NO CROSSING in the given g1 windows -- widen or move them")
        return None

    # Score every candidate against the screens the crossing did NOT use, and do
    # it by RE-EVALUATING both legs at the candidate's g1 rather than
    # interpolating the scan. Interpolation is not good enough here: near a
    # waist sigma(g1) has a corner, and on a coarse grid the interpolated
    # residual came out 17 % where the true value is 1 %. Two more evaluations
    # per candidate is a small price for a number that means something.
    def nearest_seed(rows, g):
        """The (g2, g3) of the nearest scan point that WAS round, or `g_seed`.

        A candidate sits between scan points, so its neighbours' answer is the
        best seed available -- without it the re-evaluation can land on a
        different branch and hand back a nonsense working point (seen: a
        100 1/m^2 quad).
        """
        ok = [r for r in rows if np.isfinite(r["sigma"][0])]
        if not ok:
            return g_seed
        near = min(ok, key=lambda r: abs(r["g1"] - g))
        return (near["g2"], near["g3"])

    crossings, solved = [], []
    for gf, gb, sa, sc_ in raw:
        cand = match_at(dist, geom, target, gf, gb, sc=sc,
                        sc_knobs=sc_knobs, screen_pair=(i, j),
                        round_tol=round_tol,
                        unit_step=unit_step, mesh=mesh,
                        g_seed=nearest_seed(fwd, gf),
                        g_seed_bwd=nearest_seed(bwd_rows, gb), bwd=bwd)
        crossings.append(dict(g1_fwd=gf, g1_bwd=gb, sigma=(sa, sc_),
                              residual=cand.screen_residual))
        solved.append(cand)
    if pick == "consistent":
        if free:
            # nan means a leg had no round solution there -- unusable, never
            # preferred; np.argmin would have returned the first nan instead
            score = [c["residual"] if np.isfinite(c["residual"]) else np.inf
                     for c in crossings]
            if all(np.isinf(v) for v in score):
                pick = 0
                if verbose:
                    print("  [warn] no candidate could be checked -- every one "
                          "has a leg with no round solution. Taking the first.")
            else:
                pick = int(np.argmin(score))
        else:
            pick = 0
            if verbose and len(crossings) > 1:
                print("  [warn] screen_pair uses every screen, so the crossings "
                      "cannot be checked -- taking the first. Add a screen.")
    if verbose:
        for n, c in enumerate(crossings):
            res = ("   free screens differ by "
                   f"{c['residual']*100:6.2f} %" if np.isfinite(c["residual"])
                   else "   (UNUSABLE: a leg has no round solution here)")
            print(f"  crossing {n}{' <- picked' if n == pick else '':>10s}: "
                  f"g1_fwd={c['g1_fwd']:+.4f}  g1_bwd={c['g1_bwd']:+.4f} T/m   "
                  f"sigma=({c['sigma'][0]*1e3:.4f}, {c['sigma'][1]*1e3:.4f}) mm"
                  + res)
    match = solved[pick]
    match.screen_sigma = crossings[pick]["sigma"]
    match.crossings, match.picked = crossings, int(pick)
    match.fwd_rows, match.bwd_rows = fwd, bwd_rows
    return match


# ------------------------------ the full line ------------------------------

def build_lattice(geom, t1_g, t2_g, energy_gev):
    """The whole section as an ocelot lattice, in physical order.

    `t1_g` / `t2_g` are (Q1, Q2, Q3) gradients [T/m] as `TwoTripletMatch`
    reports them -- no reordering needed at the call site.
    """
    b = brho(energy_gev)
    seq = [oc.Marker(eid="START"), oc.Drift(l=geom.t1_drifts[0])]
    for i in range(3):
        seq.append(oc.Quadrupole(l=geom.t1_lq[i], k1=t1_g[i] / b,
                                 eid=f"T1.Q{i+1}"))
        if i < 2:
            seq.append(oc.Drift(l=geom.t1_drifts[i + 1]))
    seq.append(oc.Drift(l=geom.gap))
    for i in range(3):
        seq.append(oc.Quadrupole(l=geom.t2_lq[i], k1=t2_g[i] / b,
                                 eid=f"T2.Q{i+1}"))
        seq.append(oc.Drift(l=geom.t2_drifts[i]))
    return oc.MagneticLattice(seq + [oc.Marker(eid="END")])


def track_e2e(match, dist, sc=True, unit_step=SC_UNIT_STEP, mesh=SC_MESH,
              return_particles=False):
    """Track `dist` through the matched section; fills in `match.exit_twiss`.

    This is the validation, so `sc=True` is the setting that means anything;
    `sc=False` gives the linear reference on the same lattice. Either way the
    twiss output is sampled at `unit_step` -- see `_Sampler` for why that needs
    saying. Returns the Ocelot twiss list, or ``(tws, exit_particles)`` when
    ``return_particles=True``, and mutates `match` so its `summary()` and
    `bmag()` report the delivered Twiss.
    """
    lat = build_lattice(match.geom, match.t1_g, match.t2_g, match.energy_gev)
    navi = oc.Navigator(lat, unit_step=unit_step)
    navi.add_physics_proc(
        oc.SpaceCharge(1, nmesh_xyz=list(mesh)) if sc else _Sampler(),
        lat.sequence[0], lat.sequence[-1])
    tws, pa_out = oc.track(lat, as_ocelot(dist), navi, print_progress=False)
    t = tws[-1]
    match.exit_twiss = (t.beta_x, t.alpha_x, t.beta_y, t.alpha_y)
    return (tws, pa_out) if return_particles else tws
