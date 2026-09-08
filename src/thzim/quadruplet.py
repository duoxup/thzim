#!/usr/bin/env python3
r"""Quadruplet matching section: four quads against four exit Twiss numbers.

    START  d_in  Q1  d_inter  Q2  d_inter  Q3  d_inter  Q4  d_out  END

One of the two ways this package matches a beam onto a target Twiss; the other
is `thzim.two_triplet`. This one is the MODEL route: four knobs against the four
exit Twiss numbers (beta_x, alpha_x, beta_y, alpha_y), a square problem solved
against the transfer matrix. In the design of record it is the superradiant
branch's M1 (launch = the P0 Twiss, target = the chicane entrance Twiss from
`thzim.chicane`) and the final focus M3 of BOTH branches onto the P3 target
(with per-gap spacing for the superradiant M3, see `QuadrupletGeom`).

Two layers, and they are not interchangeable:

  solve_linear_match(geom, launch, target, E) -> QuadrupletMatch
      exact transfer-matrix least squares from many starts, keeping the
      min-peak-beta solution.
  sc_rematch(match, dist) -> QuadrupletMatch
      correct the four k1 under SPACE CHARGE, starting from that solution.

## Why the linear layer needs many starts

Four quads on four Twiss constraints is square but strongly non-linear, and it
has SEVERAL exact solutions with wildly different beam sizes inside the section.
Local least squares lands on whichever one the seed is nearest. So the solver
runs 9 hand-picked antisymmetric seeds plus `n_random` uniform ones, keeps only
those that converge to a true match (`2 * cost < 1e-8`, i.e. the residual is at
round-off), and among those returns the smallest `max(peak beta_x, peak
beta_y)`. That is the same aperture-control criterion `thzim.dogleg` uses to
pick its free inner-pair strength.

## Why the SC layer is a damped Newton with a LINEAR Jacobian

Two choices in `sc_rematch` carry all the robustness:

* **The residual is RELATIVE in the betas**, `(beta/beta_target - 1)`, and
  absolute in the alphas. Targets here routinely mix decades -- a 6 mm vertical
  waist next to a 10 m horizontal beta -- and an absolute residual would let the
  large one swallow the small one.
* **The residual is measured by tracking, the Jacobian is not.** The Jacobian
  comes from the noise-free linear optics, by central differences, recomputed at
  the current k every step (which costs nothing). A finite-difference Jacobian
  taken on the SC track would be differencing statistical noise.

On top of that: backtracking (lambda = 1, 1/2, 1/4, 1/10, 1/25; a step that
raises the cost is rejected, a crashed track counts as infinite cost), and an
outer TARGET-SHIFTING stage for shifts that start outside the Newton basin --
re-aim the globally solved linear match at a shifted target so that "linear
target plus SC shift" lands on the true one, betas multiplicatively, alphas
additively. The solver returns the best k it saw and says so if `tol` was missed.

## The linear layer carries no energy

Ocelot's `k1` is a geometric strength and a quadrupole map depends only on
(k1, L), so `solve_linear_match` returns the same k1 at any `energy_gev`.
Energy enters in exactly two places: `QuadrupletMatch.gradients` (the k1 -> T/m
conversion) and the space-charge track. Same situation as `thzim.dogleg`.

Beams are partdist distributions here as everywhere in the package; the SC layer
converts to ocelot once, internally. Building a beam is the caller's job -- this
module matches beams, it does not manufacture them.
"""

import copy
from dataclasses import dataclass

import numpy as np
import ocelot as oc
from ocelot.cpbd.optics import lattice_transfer_map
from ocelot.cpbd.physics_proc import PhysProc
from scipy.optimize import least_squares

from thzim.maps import propagate_twiss
from thzim.triplet import as_ocelot
from thzim.utils import brho

__all__ = [
    "QuadrupletGeom", "QuadrupletMatch", "build_lattice", "bmag",
    "exit_twiss_linear", "peak_beta", "solve_linear_match", "track_match",
    "sc_rematch",
]


# ------------------------------- the geometry -------------------------------

@dataclass
class QuadrupletGeom:
    """Four-quad matching section (all lengths in m).

    START d_in Q1 d_inter Q2 d_inter Q3 d_inter Q4 d_out END. All four quads
    share one length -- this is a matching section, not a footprint-constrained
    insertion like `ChicaneGeom` or `DoglegGeom`, so its geometry is set by
    what fits between two interface planes.

    `d_inter` is one spacing shared by the three gaps (the M1 station) or a
    tuple of three, (Q1-Q2, Q2-Q3, Q3-Q4), when the four powered quads are not
    evenly spaced. The superradiant branch's M3 is the case: T1's last quad
    plus the T2 triplet across the 0.92 m switch region, with T1's first two
    quads switched off, so the gaps are (0.92, 0.30, 0.30) and the dead quads
    fold into `d_in`.
    """

    lq: float = 0.10              # quad magnetic length
    d_in: float = 0.30            # section start -> Q1
    d_inter: object = 0.30        # between adjacent quads: one value or three
    d_out: float = 0.40           # Q4 -> section end

    @property
    def d_inters(self):
        """The three inter-quad drifts (Q1-Q2, Q2-Q3, Q3-Q4) [m]."""
        if np.ndim(self.d_inter) == 0:
            return (float(self.d_inter),) * 3
        if len(self.d_inter) != 3:
            raise ValueError("d_inter must be one spacing or a tuple of three")
        return tuple(float(d) for d in self.d_inter)

    @property
    def length(self):
        return self.d_in + 4.0 * self.lq + sum(self.d_inters) + self.d_out

    def element_spans(self):
        """(s_start, s_end, name) of each quadrupole, s from the START marker."""
        spans, s = [], self.d_in
        for i, gap in enumerate(self.d_inters + (0.0,)):
            spans.append((s, s + self.lq, f"Q{i + 1}"))
            s += self.lq + gap
        return spans

    def summary(self):
        gaps = self.d_inters
        d_inter = (f"{gaps[0]:.3f}" if len(set(gaps)) == 1
                   else "(" + ", ".join(f"{d:.3f}" for d in gaps) + ")")
        return (f"Quadruplet  (Lq={self.lq:.3f}, d_in={self.d_in:.3f}, "
                f"d_inter={d_inter}, d_out={self.d_out:.3f} m)\n"
                f"  length={self.length:.4f} m")


def build_lattice(geom, k1s, sliced=False, nsl=20):
    """The quadruplet as an Ocelot MagneticLattice, with START / END markers.

    Element instances are independent (safe for tracking); `sliced=True` cuts
    every element into `nsl` pieces, which is what makes peak-beta detection
    smooth rather than element-granular.
    """
    if sliced:
        def D(L):
            return [oc.Drift(l=L / nsl) for _ in range(nsl)] if L > 0 else []

        def Q(k, eid):
            return [oc.Quadrupole(l=geom.lq / nsl, k1=k, eid=eid)
                    for _ in range(nsl)]
    else:
        def D(L):
            return [oc.Drift(l=L)] if L > 0 else []

        def Q(k, eid):
            return [oc.Quadrupole(l=geom.lq, k1=k, eid=eid)]

    g12, g23, g34 = geom.d_inters
    seq = ([oc.Marker(eid="START")] + D(geom.d_in)
           + Q(k1s[0], "Q1") + D(g12)
           + Q(k1s[1], "Q2") + D(g23)
           + Q(k1s[2], "Q3") + D(g34)
           + Q(k1s[3], "Q4") + D(geom.d_out)
           + [oc.Marker(eid="END")])
    return oc.MagneticLattice(seq)


def bmag(beta, alpha, beta0, alpha0=0.0):
    """Twiss mismatch amplification against (beta0, alpha0).

    1.0 is matched, and the value is the effective emittance growth a mismatched
    beam suffers once it filaments -- which is why it, rather than the raw Twiss
    difference, is the number to judge a match by.
    """
    g0 = (1.0 + alpha0**2) / beta0
    g = (1.0 + alpha**2) / beta
    return 0.5 * (beta0 * g - 2.0 * alpha0 * alpha + g0 * beta)


# ---------------------------- linear optics core ----------------------------

def exit_twiss_linear(geom, k1s, launch, energy_gev):
    """Exit (beta_x, alpha_x, beta_y, alpha_y) from the transfer matrix.

    No beam involved, and no energy either -- see the module docstring.
    """
    R = lattice_transfer_map(build_lattice(geom, list(k1s)), energy_gev)
    bx, ax = propagate_twiss(R[0:2, 0:2], launch[0], launch[1])
    by, ay = propagate_twiss(R[2:4, 2:4], launch[2], launch[3])
    return (bx, ax, by, ay)


def peak_beta(geom, k1s, launch, energy_gev, nsl=20):
    """(peak beta_x, peak beta_y) through the section, on a sliced lattice.

    Sliced so that a peak falling BETWEEN elements is not missed -- the same
    precaution `thzim.dogleg.solve_entrance_twiss` takes.
    """
    lat = build_lattice(geom, list(k1s), sliced=True, nsl=nsl)
    t0 = oc.Twiss()
    t0.E = energy_gev
    t0.beta_x, t0.alpha_x, t0.beta_y, t0.alpha_y = launch
    tws = oc.twiss(lat, t0)
    return max(t.beta_x for t in tws), max(t.beta_y for t in tws)


def _residual(geom, k1s, launch, target, energy_gev):
    e = exit_twiss_linear(geom, k1s, launch, energy_gev)
    return [e[i] - target[i] for i in range(4)]


# ------------------------------- the solution -------------------------------

@dataclass
class QuadrupletMatch:
    """Result of a quadruplet match, linear or SC-corrected."""

    geom: QuadrupletGeom
    k1s: list                      # the four quad k1 [1/m^2]
    energy_gev: float
    launch: tuple                  # (beta_x, alpha_x, beta_y, alpha_y) at START
    target: tuple                  # (beta_x, alpha_x, beta_y, alpha_y) wanted
    lattice: object                # oc.MagneticLattice
    exit_twiss: tuple              # what the section actually delivers
    peak_beta_x: float
    peak_beta_y: float
    sc_matched: bool = False
    seed_exit_twiss: tuple = None  # SC re-match only: the exit under SC at the
                                   # LINEAR k1s, i.e. the pre-correction baseline

    @property
    def gradients(self):
        """The four gradients [T/m] -- the only place energy enters."""
        b = brho(self.energy_gev)
        return [k * b for k in self.k1s]

    def bmag(self):
        bx, ax, by, ay = self.exit_twiss
        return (bmag(bx, ax, self.target[0], self.target[1]),
                bmag(by, ay, self.target[2], self.target[3]))

    def summary(self):
        bmx, bmy = self.bmag()
        head = "SC-matched" if self.sc_matched else "linear"
        lines = [
            f"Quadruplet match ({head}, E={self.energy_gev*1e3:.1f} MeV, "
            f"length={self.geom.length:.3f} m)",
            f"  launch  bx={self.launch[0]:.2f} ax={self.launch[1]:+.2f}  "
            f"by={self.launch[2]:.2f} ay={self.launch[3]:+.2f}",
            f"  target  bx={self.target[0]:.3f} ax={self.target[1]:+.2f}  "
            f"by={self.target[2]:.3f} ay={self.target[3]:+.2f}",
            f"  exit    bx={self.exit_twiss[0]:.3f} ax={self.exit_twiss[1]:+.3f}"
            f"  by={self.exit_twiss[2]:.3f} ay={self.exit_twiss[3]:+.3f}   "
            f"Bmag={bmx:.3f}/{bmy:.3f}"]
        for i, (k, g) in enumerate(zip(self.k1s, self.gradients)):
            lines.append(f"  Q{i+1}  k1={k:+9.3f} 1/m^2   g={g:+8.3f} T/m")
        lines.append(f"  peak beta_x={self.peak_beta_x:.2f}  "
                     f"beta_y={self.peak_beta_y:.2f} m")
        return "\n".join(lines)


def solve_linear_match(geom, launch, target, energy_gev=0.0395, n_random=40,
                       seed=0):
    """Match the four quads to `target`, exactly, from many starts.

    Keeps the converged solution with the smallest peak beta; see the module
    docstring for why the multi-start is not optional. Raises if the geometry
    admits no solution at all. `energy_gev` does not affect the k1 -- it only
    labels the gradients.

    The tie-break is DEGENERATE when the peak beta sits at the exit: every
    exact solution then shares peak beta_x = the target (the OP3/OP4 M1 case,
    30 m), and which of them wins can change with the last decimals of the
    launch Twiss. All of them are valid matches; do not expect the k1 to be
    stable under small changes of the inputs there.
    """
    rng = np.random.default_rng(seed)
    seeds = [[10, -10, 10, -10], [-10, 10, -10, 10], [8, -6, 6, -8],
             [-8, 6, -6, 8], [15, -15, 15, -15], [5, -12, 12, -5],
             [20, -20, 10, -10], [-15, 20, -20, 15], [30, -30, 20, -10]]
    seeds += [list(rng.uniform(-40, 40, 4)) for _ in range(n_random)]
    best = None
    for s in seeds:
        sol = least_squares(
            lambda k: _residual(geom, k, launch, target, energy_gev),
            s, method="lm", max_nfev=4000)
        if 2.0 * sol.cost > 1e-8:               # not an exact match; discard
            continue
        pbx, pby = peak_beta(geom, sol.x, launch, energy_gev)
        if best is None or max(pbx, pby) < max(best[1], best[2]):
            best = (list(sol.x), pbx, pby)
    if best is None:
        raise RuntimeError(
            "solve_linear_match: no four-quad solution for this geometry")
    k1s, pbx, pby = best
    return QuadrupletMatch(
        geom=geom, k1s=k1s, energy_gev=energy_gev, launch=tuple(launch),
        target=tuple(target), lattice=build_lattice(geom, k1s),
        exit_twiss=exit_twiss_linear(geom, k1s, launch, energy_gev),
        peak_beta_x=pbx, peak_beta_y=pby)


# --------------------------- tracking and SC layer ---------------------------


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


def _track_pa(lattice, pa, sc=True, unit_step=0.05, nmesh=(63, 63, 63),
              return_particles=False):
    """Track an Ocelot ParticleArray, optionally returning the exit particles."""
    navi = oc.Navigator(lattice)
    navi.unit_step = unit_step
    if sc:
        proc = oc.SpaceCharge()
        proc.nmesh_xyz = list(nmesh)
        proc.step = 1
    else:
        proc = _Sampler()          # sampling only; see the class docstring
    navi.add_physics_proc(proc, lattice.sequence[0], lattice.sequence[-1])
    tws, pa_out = oc.track(lattice, copy.deepcopy(pa), navi,
                           print_progress=False)
    t = tws[-1]
    result = ((t.beta_x, t.alpha_x, t.beta_y, t.alpha_y), tws)
    return (*result, pa_out) if return_particles else result


def track_match(lattice, dist, sc=True, unit_step=0.05, nmesh=(63, 63, 63),
                return_particles=False):
    """Track a partdist distribution through ``lattice``.

    Converts to ocelot internally. `sc_rematch` does not go through this -- it
    converts once and reuses the array across its many tracks. Returns
    ``(exit_twiss, tws)`` normally and appends the exit Ocelot ParticleArray
    when ``return_particles=True``.
    """
    return _track_pa(lattice, as_ocelot(dist), sc=sc, unit_step=unit_step,
                     nmesh=nmesh, return_particles=return_particles)


def _rel_residual(twiss, target):
    """Relative in the betas, absolute in the alphas -- see the module docstring."""
    return np.array([twiss[0] / target[0] - 1.0, twiss[1] - target[1],
                     twiss[2] / target[2] - 1.0, twiss[3] - target[3]])


def _rel_jacobian(geom, k, launch, target, energy_gev, h=0.05):
    """d(relative residual)/dk from the NOISE-FREE linear optics, central diff."""
    J = np.zeros((4, 4))
    for j in range(4):
        kp, km = k.copy(), k.copy()
        kp[j] += h
        km[j] -= h
        rp = _rel_residual(exit_twiss_linear(geom, list(kp), launch, energy_gev),
                           target)
        rm = _rel_residual(exit_twiss_linear(geom, list(km), launch, energy_gev),
                           target)
        J[:, j] = (rp - rm) / (2.0 * h)
    return J


def sc_rematch(match, dist, n_iter=8, tol=5e-4, unit_step=0.05,
               nmesh=(63, 63, 63), verbose=True):
    """Re-match the four k1 under space charge, from a linear QuadrupletMatch.

    Damped Newton: relative residual from an SC track of `dist`, Jacobian from
    the linear optics, backtracking on the step, plus an outer target-shifting
    stage for shifts outside the Newton basin. See the module docstring for why
    each of those is there. Returns the best k it saw, and warns if `tol` was
    not reached rather than raising -- a best-effort match is still usable and
    its Bmag says how usable.

    `dist` is a partdist distribution; it is converted once, here.
    """
    geom, launch, target, E = (match.geom, match.launch, match.target,
                               match.energy_gev)
    pa = as_ocelot(dist)

    def tracked(k):
        try:
            tw, _ = _track_pa(build_lattice(geom, list(k)), pa, sc=True,
                              unit_step=unit_step, nmesh=nmesh)
        except Exception:
            return None, None, np.inf
        r = _rel_residual(tw, target)
        return tw, r, float(r @ r)

    k = np.array(match.k1s, float)
    exit_tw, r, cost = tracked(k)
    if exit_tw is None:
        raise RuntimeError("sc_rematch: SC tracking failed at the linear k1s")
    seed_exit = exit_tw
    best = (cost, k.copy(), exit_tw)
    if verbose:
        print(f"  Newton 0: cost={cost:.3e}  beta_x={exit_tw[0]:.3f} "
              f"beta_y={exit_tw[2]:.4f}  k=["
              + ", ".join(f"{x:+.1f}" for x in k) + "]")

    # outer target-shifting stage: aim the (globally solved) LINEAR match at a
    # shifted target so that "linear target + SC shift" lands on the true one.
    # Betas shift multiplicatively so they stay positive, alphas additively.
    tgt_eff = np.array(target, float)
    for outer in range(4):
        if cost < 0.5:
            break
        tw = best[2]
        tgt_eff[0] *= target[0] / max(tw[0], 1e-9)
        tgt_eff[2] *= target[2] / max(tw[2], 1e-9)
        tgt_eff[1] += target[1] - tw[1]
        tgt_eff[3] += target[3] - tw[3]
        try:
            lin = solve_linear_match(geom, launch, tuple(tgt_eff),
                                     energy_gev=E, n_random=15, seed=outer)
        except RuntimeError:
            break
        kt = np.array(lin.k1s, float)
        twt, rt, ct = tracked(kt)
        if verbose:
            bx = twt[0] if twt else float("nan")
            by = twt[2] if twt else float("nan")
            print(f"  target-shift {outer}: eff betas=({tgt_eff[0]:.3f}, "
                  f"{tgt_eff[2]:.5f})  cost={ct:.3e}  beta_x={bx:.3f} "
                  f"beta_y={by:.4f}")
        if ct < best[0]:
            best = (ct, kt.copy(), twt)
            k, exit_tw, r, cost = kt, twt, rt, ct
        else:
            break

    for it in range(1, n_iter + 1):
        if cost < tol:
            break
        J = _rel_jacobian(geom, k, launch, target, E)
        try:
            dk = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            break
        accepted = False
        for lam in (1.0, 0.5, 0.25, 0.1, 0.04):
            kt = k - lam * dk
            twt, rt, ct = tracked(kt)
            if ct < cost:
                k, exit_tw, r, cost = kt, twt, rt, ct
                accepted = True
                if cost < best[0]:
                    best = (cost, k.copy(), exit_tw)
                if verbose:
                    print(f"  Newton {it}: cost={cost:.3e} (lam={lam:g})  "
                          f"beta_x={exit_tw[0]:.3f} beta_y={exit_tw[2]:.4f}  "
                          "k=[" + ", ".join(f"{x:+.1f}" for x in k) + "]")
                break
        if not accepted:
            if verbose:
                print(f"  Newton {it}: no step decreased the cost -- stopping "
                      f"at best-effort")
            break

    cost, k, exit_tw = best
    if cost >= tol and verbose:
        print(f"  sc_rematch: tol not reached (best cost={cost:.3e}) -- "
              f"returning best k")
    k1s = list(k)
    pbx, pby = peak_beta(geom, k1s, launch, E)
    return QuadrupletMatch(
        geom=geom, k1s=k1s, energy_gev=E, launch=launch, target=target,
        lattice=build_lattice(geom, k1s), exit_twiss=exit_tw,
        peak_beta_x=pbx, peak_beta_y=pby, sc_matched=True,
        seed_exit_twiss=seed_exit)
