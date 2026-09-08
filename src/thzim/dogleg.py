#!/usr/bin/env python3
r"""Achromatic dogleg design: geometry, achromat quad solution, entrance Twiss.

A 2-family mirror-symmetric achromatic dogleg:

    START  B1(+t)  d1  QO  d2  QI  d3 | C | d3  QI  d2  QO  d1  B2(-t)  END

* QO -- outer pair (shared strength ko): the DISPERSION knob. The achromat
  condition Dx = Dx' = 0 at the exit reduces -- by mirror symmetry plus
  antisymmetric dipole forcing -- to the single condition Dx(C) = 0, which pins
  ko.
* QI -- inner pair (shared strength ki): the VERTICAL knob. The achromat does
  NOT constrain it, so ki is an INPUT here; ``design_dogleg`` picks it.

The inter-dipole straight L_gap is fixed by the transverse-offset closure
Delta_x = 2 rho (1 - cos t) + L_gap sin t, and the innermost drift d3 fills
L_gap/2.

Three layers:

  API 1  solve_achromat_quads(geom, ki) -> DoglegLattice
         given geometry + inner-pair ki, solve outer-pair ko for the achromat.
  API 2  solve_entrance_twiss(dl) -> EntranceTwiss
         for a FIXED lattice, solve the target entrance Twiss (x, y decoupled).
  driver design_dogleg(geom, ki_scan, emit_ratio) -> (DoglegLattice, EntranceTwiss)
         scan ki, loop API 1 + API 2, keep the roundest (min-peak) solution.

NOTHING HERE READS A PARTICLE DISTRIBUTION. Both solves are pure linear optics:
ko, the entrance Twiss and the peak betas come out bit-identical at 15.4, 39.4
and 100 MeV, because Ocelot's k1 [1/m^2] and the bend angle are geometric.
`energy_gev` only sets the R56 velocity term and the k1 -> T/m conversion. The
one place a beam property may enter is `emit_ratio` in the driver, and it is
optional -- see below.

Sizing helpers (vectorised, angles in RADIANS) sit at module level so the
tools/ maps can share them. rho_for_r56() is the geometry-only closed form,

    R56_z = -2 rho (theta - sin theta)   ->   rho = |R56_z| / (2 (theta - sin theta))

which drops the velocity term -L_tot/(beta gamma)^2. That approximation is
accepted by design, since the dogleg only needs an approximate R56 (its peak
current is trimmed afterwards with the injector chirp). Use
rho_for_r56_exact() when the velocity term is wanted.

Migrated from the earlier dogleg design implementation; see MIGRATION.md for
the differences.
"""

from dataclasses import dataclass
from math import radians, sin

import numpy as np
import ocelot as oc
from ocelot.cpbd.match import match
from ocelot.cpbd.optics import lattice_transfer_map
from scipy.optimize import brentq, minimize_scalar

from thzim.utils import M_E_GEV, brho

__all__ = [
    "M_E_GEV", "brho",
    "r56_z_geometric", "rho_for_r56", "rho_for_r56_exact", "dipole_offset",
    "offset", "gap_for_offset", "arc_length", "z_footprint", "field",
    "DoglegGeom", "DoglegLattice", "EntranceTwiss",
    "build_lattice", "element_spans", "solve_achromat_quads",
    "solve_entrance_twiss", "twiss_along", "design_dogleg",
]

# M_E_GEV / brho live in thzim.utils so that every module shares one definition;
# they are re-exported here (see __all__) because this module's public API has
# always carried them.


# --------------------------- geometry (vectorised) ---------------------------
# theta is in RADIANS throughout this block; DoglegGeom below takes degrees.

def r56_z_geometric(theta, rho):
    """R56 [m], z convention (< 0 for a dogleg), geometry only."""
    return -2.0 * rho * (theta - np.sin(theta))


def rho_for_r56(r56_z_target, theta):
    """Bend radius [m] delivering the requested R56 -- the closed-form inverse.

    Geometry only: the velocity term is dropped, which is what removes L_gap
    (and hence Delta_x) from the problem and makes this a closed form.
    """
    return np.abs(r56_z_target) / (2.0 * (theta - np.sin(theta)))


def rho_for_r56_exact(r56_z_target, theta, delta_x, energy_gev, bracket=(0.05, 5.0)):
    """Bend radius [m] including the velocity term -L_tot/(beta gamma)^2.

    L_tot depends on rho through L_gap, so this needs a scalar solve. Use it
    when the ~6 % (at 15 MeV) geometric bias actually matters; rho_for_r56() is
    the sizing-stage default.
    """
    b2g2 = (energy_gev / M_E_GEV) ** 2 - 1.0

    def residual(rho):
        l_tot = 2.0 * rho * theta + gap_for_offset(delta_x, theta, rho)
        r56_tau = 2.0 * rho * (theta - sin(theta)) - l_tot / b2g2
        return -r56_tau - r56_z_target

    return brentq(residual, *bracket)


def dipole_offset(theta, rho):
    """Transverse offset [m] from the two dipoles alone (L_gap = 0)."""
    return 2.0 * rho * (1.0 - np.cos(theta))


def offset(theta, rho, l_gap):
    """Total transverse offset Delta_x [m], closed by the inter-dipole straight."""
    return dipole_offset(theta, rho) + l_gap * np.sin(theta)


def gap_for_offset(delta_x, theta, rho):
    """Straight length [m] between the dipoles that closes Delta_x."""
    return (delta_x - dipole_offset(theta, rho)) / np.sin(theta)


def arc_length(theta, rho):
    """Dipole arc length [m]."""
    return rho * theta


def z_footprint(theta, rho, l_gap):
    """Longitudinal extent [m] of the two dipoles plus the straight."""
    return 2.0 * rho * np.sin(theta) + l_gap * np.cos(theta)


def field(rho, energy_gev):
    """Dipole field [T] at the given total energy."""
    return brho(energy_gev) / rho


# ------------------------------- the geometry -------------------------------

@dataclass
class DoglegGeom:
    """Offset-constrained 2-family dogleg (lengths in m, angle in deg)."""

    theta_deg: float
    rho: float
    delta_x: float
    lq_o: float = 0.10            # outer-pair quad length
    lq_i: float = 0.10            # inner-pair quad length
    d1: float = 0.10              # drift  dipole -> outer quad
    d2: float = 0.10              # drift  outer quad -> inner quad
    d_in: float = 0.20            # entrance plane START -> B1
    d_out: float = 0.0            # lead-out drift B2 -> END

    @property
    def theta(self):
        return radians(self.theta_deg)

    @property
    def L_bend(self):
        return self.rho * self.theta

    @property
    def L_gap(self):
        return gap_for_offset(self.delta_x, self.theta, self.rho)

    @property
    def length(self):
        """Total lattice length START -> END [m]."""
        return self.d_in + 2.0 * self.L_bend + self.L_gap + self.d_out

    @property
    def d3(self):
        """Inner quad -> centre drift, filling L_gap/2."""
        d = self.L_gap / 2.0 - (self.d1 + self.lq_o + self.d2 + self.lq_i)
        if d <= 0.0:
            raise ValueError(
                f"infeasible geometry: d3={d:.4f} m <= 0 (quads+drifts exceed "
                f"L_gap/2={self.L_gap / 2:.3f} m); shorten quads/drifts or "
                f"lengthen L_gap.")
        return d

    @property
    def R56_z(self):
        """Geometry-only R56 [m], z convention (no velocity term)."""
        return float(r56_z_geometric(self.theta, self.rho))


def build_lattice(geom, ko, ki, sliced=False, nsl=20):
    """Mirror-symmetric 2-family dogleg with START / C (centre) / END markers.

    Element instances are independent (safe for tracking); set sliced=True to
    get a finely-cut lattice, which is what makes peak-beta detection smooth.
    """
    Lb, d3 = geom.L_bend, geom.d3
    if sliced:
        def D(L):
            return [oc.Drift(l=L / nsl) for _ in range(nsl)] if L > 0 else []

        def B(a):
            return [oc.SBend(l=Lb / nsl, angle=a / nsl, eid="B")
                    for _ in range(nsl)]

        def Q(k, e, lq):
            return [oc.Quadrupole(l=lq / nsl, k1=k, eid=e) for _ in range(nsl)]
    else:
        def D(L):
            return [oc.Drift(l=L)] if L > 0 else []

        def B(a):
            return [oc.SBend(l=Lb, angle=a, eid="B")]

        def Q(k, e, lq):
            return [oc.Quadrupole(l=lq, k1=k, eid=e)]

    seq = ([oc.Marker(eid="START")] + D(geom.d_in)
           + B(+geom.theta) + D(geom.d1) + Q(ko, "QO", geom.lq_o)
           + D(geom.d2) + Q(ki, "QI", geom.lq_i) + D(d3)
           + [oc.Marker(eid="C")]
           + D(d3) + Q(ki, "QI", geom.lq_i) + D(geom.d2) + Q(ko, "QO", geom.lq_o)
           + D(geom.d1) + B(-geom.theta)
           + D(geom.d_out) + [oc.Marker(eid="END")])
    return oc.MagneticLattice(seq)


def element_spans(geom):
    """(s_start, s_end, kind) of every bend and quad along the dogleg.

    kind is "bend", "QO" or "QI"; drifts are skipped. Layout knowledge, so it
    lives with the geometry rather than in whichever script draws it.
    """
    spans, s = [], geom.d_in
    for length, kind in ((geom.L_bend, "bend"), (geom.d1, None),
                         (geom.lq_o, "QO"), (geom.d2, None),
                         (geom.lq_i, "QI"), (geom.d3, None),
                         (geom.d3, None), (geom.lq_i, "QI"),
                         (geom.d2, None), (geom.lq_o, "QO"),
                         (geom.d1, None), (geom.L_bend, "bend")):
        if kind is not None:
            spans.append((s, s + length, kind))
        s += length
    return spans


# ------------------------ API 1: the achromat solution ------------------------

@dataclass
class DoglegLattice:
    """Result of API 1: a fully-defined achromatic dogleg."""

    geom: DoglegGeom
    ki: float                     # inner-pair k1 [1/m^2]  (input)
    ko: float                     # outer-pair k1 [1/m^2]  (solved for achromat)
    energy_gev: float
    lattice: oc.MagneticLattice   # START / C / END markers, for API 2
    Dx_exit: float                # residual dispersion at exit (-> 0 if closed)
    Dxp_exit: float
    R56_z: float                  # z-convention R56 [m], from the exact map

    @property
    def g_o(self):
        """Outer-pair gradient [T/m]."""
        return self.ko * brho(self.energy_gev)

    @property
    def g_i(self):
        """Inner-pair gradient [T/m]."""
        return self.ki * brho(self.energy_gev)

    @property
    def achromatic(self):
        return abs(self.Dx_exit) < 1e-4 and abs(self.Dxp_exit) < 1e-4

    def summary(self):
        g = self.geom
        return (
            f"Achromatic dogleg  (theta={g.theta_deg:.2f} deg, rho={g.rho:.4f} m, "
            f"Delta_x={g.delta_x:.3f} m, E={self.energy_gev*1e3:.1f} MeV)\n"
            f"  geometry: L_bend={g.L_bend:.3f}  L_gap={g.L_gap:.3f}  "
            f"d3={g.d3:.3f}  length={g.length:.3f} m\n"
            f"            (d_in={g.d_in}, d1={g.d1}, lq_o={g.lq_o}, d2={g.d2}, "
            f"lq_i={g.lq_i}, d_out={g.d_out})\n"
            f"  QO (outer, disp.): k1={self.ko:+.4f} 1/m^2   "
            f"g={self.g_o:+.4f} T/m\n"
            f"  QI (inner, vert.): k1={self.ki:+.4f} 1/m^2   "
            f"g={self.g_i:+.4f} T/m   (input)\n"
            f"  exit Dx={self.Dx_exit:+.2e} m  Dxp={self.Dxp_exit:+.2e}  -> "
            f"{'ACHROMATIC' if self.achromatic else 'NOT closed'}\n"
            f"  R56_z={self.R56_z:+.4f} m  "
            f"(geometry-only {self.geom.R56_z:+.4f} m)")


def solve_achromat_quads(geom, ki, energy_gev=0.0395, ko_seed=40.0):
    """API 1: solve the OUTER-pair ko making the dogleg achromatic, given ki.

    The match runs on Dx(C) = 0 with a SHARED QO instance appearing on both
    sides of the centre, so there is exactly one match variable. The returned
    lattice is then rebuilt with independent instances (tracking-safe).
    """
    Lb, d3 = geom.L_bend, geom.d3
    qo = oc.Quadrupole(l=geom.lq_o, k1=ko_seed, eid="QO")
    qi = oc.Quadrupole(l=geom.lq_i, k1=ki, eid="QI")
    mkc = oc.Marker(eid="C")
    seq = [oc.Marker(eid="START"), oc.SBend(l=Lb, angle=+geom.theta, eid="B1"),
           oc.Drift(l=geom.d1), qo, oc.Drift(l=geom.d2), qi, oc.Drift(l=d3), mkc,
           oc.Drift(l=d3), qi, oc.Drift(l=geom.d2), qo, oc.Drift(l=geom.d1),
           oc.SBend(l=Lb, angle=-geom.theta, eid="B2"), oc.Marker(eid="END")]
    lat_m = oc.MagneticLattice(seq)
    t0 = oc.Twiss()
    t0.beta_x = t0.beta_y = 1.0
    t0.E = energy_gev
    match(lat_m, {mkc: {"Dx": 0.0}}, [qo], t0, verbose=False, max_iter=2000)
    ko = qo.k1

    lat = build_lattice(geom, ko, ki, sliced=False)
    tw = oc.twiss(lat, t0)
    r56_z = -lattice_transfer_map(lat, energy_gev)[4, 5]
    return DoglegLattice(geom=geom, ki=ki, ko=ko, energy_gev=energy_gev,
                         lattice=lat, Dx_exit=tw[-1].Dx, Dxp_exit=tw[-1].Dxp,
                         R56_z=r56_z)


# ------------------------ API 2: the entrance Twiss ------------------------

@dataclass
class EntranceTwiss:
    """Target entrance Twiss, to be delivered by the upstream match."""

    beta_x: float
    alpha_x: float
    beta_y: float
    alpha_y: float
    beta_cx: float            # centre waist beta_x(C)
    beta_cy: float            # centre waist beta_y(C)
    peak_beta_x: float        # min-peak beta_x over the dogleg
    peak_beta_y: float

    def as_tuple(self):
        return (self.beta_x, self.alpha_x, self.beta_y, self.alpha_y)

    def peak_score(self, emit_ratio=1.0):
        """Worst-plane peak beam size squared, in units of eps_x.

        sigma_x^2 = eps_x beta_x and sigma_y^2 = eps_y beta_y, so with
        emit_ratio = eps_y/eps_x both planes compare as (beta_x, r beta_y).
        emit_ratio = 1.0 reduces to max(peak_beta_x, peak_beta_y), i.e. the
        purely optical, distribution-free criterion.
        """
        return max(self.peak_beta_x, emit_ratio * self.peak_beta_y)

    def summary(self):
        return (
            "Entrance Twiss (centre-waist min-peak; deliver this with the "
            "upstream match)\n"
            f"  x: beta={self.beta_x:7.3f} m  alpha={self.alpha_x:+7.3f}   "
            f"(centre beta={self.beta_cx:.3f} m, peak beta={self.peak_beta_x:.2f} m)\n"
            f"  y: beta={self.beta_y:7.3f} m  alpha={self.alpha_y:+7.3f}   "
            f"(centre beta={self.beta_cy:.3f} m, peak beta={self.peak_beta_y:.2f} m)")


def _launch_from_betac(beta_c, M2):
    """Entrance (beta0, alpha0) from a centre waist beta_c back through M2."""
    minv = np.linalg.inv(M2)
    sig = minv @ np.array([[beta_c, 0.0], [0.0, 1.0 / beta_c]]) @ minv.T
    return sig[0, 0], -sig[0, 1]


def _peak_beta(beta_c, M2, plane, lat_s, energy):
    b0, a0 = _launch_from_betac(beta_c, M2)
    t0 = oc.Twiss()
    t0.E = energy
    if plane == "x":
        t0.beta_x, t0.alpha_x, t0.beta_y, t0.alpha_y = b0, a0, 5.0, 0.0
    else:
        t0.beta_x, t0.alpha_x, t0.beta_y, t0.alpha_y = 5.0, 0.0, b0, a0
    tws = oc.twiss(lat_s, t0)
    peak = max((t.beta_x if plane == "x" else t.beta_y) for t in tws)
    return peak, b0, a0


def _min_peak(M2, plane, lat_s, energy, bounds=(0.05, 25.0)):
    """Centre waist (alpha(C)=0) minimising the TRUE peak beta(s)."""
    res = minimize_scalar(lambda bc: _peak_beta(bc, M2, plane, lat_s, energy)[0],
                          bounds=bounds, method="bounded",
                          options={"xatol": 1e-3})
    peak, b0, a0 = _peak_beta(res.x, M2, plane, lat_s, energy)
    return b0, a0, peak, res.x


def solve_entrance_twiss(dl, nsl=30):
    """API 2: for a FIXED achromatic dogleg, solve the target ENTRANCE Twiss.

    The lattice has no x-y coupling, so the entrance->centre transfer matrix is
    block diagonal and the planes are solved INDEPENDENTLY. Each plane places a
    waist at the centre (alpha(C) = 0) and back-propagates it, with the centre
    beta chosen to minimise the actual peak beta through the dogleg -- measured
    on a sliced lattice, so the peak is not missed between elements.

    Distribution-free: the emittance ratio does not enter here (it cannot move
    a per-plane argmin), only the ki choice one layer up.
    """
    seq = list(dl.lattice.sequence)
    ic = [i for i, e in enumerate(seq) if getattr(e, "id", "") == "C"][0]
    M = lattice_transfer_map(oc.MagneticLattice(seq[:ic + 1]), dl.energy_gev)
    lat_s = build_lattice(dl.geom, dl.ko, dl.ki, sliced=True, nsl=nsl)
    bx0, ax0, pbx, bcx = _min_peak(M[0:2, 0:2], "x", lat_s, dl.energy_gev)
    by0, ay0, pby, bcy = _min_peak(M[2:4, 2:4], "y", lat_s, dl.energy_gev)
    return EntranceTwiss(beta_x=bx0, alpha_x=ax0, beta_y=by0, alpha_y=ay0,
                         beta_cx=bcx, beta_cy=bcy, peak_beta_x=pbx,
                         peak_beta_y=pby)


def twiss_along(dl, entrance, nsl=30):
    """Twiss through a sliced copy of the dogleg, launched from `entrance`.

    The forward check on API 2: max(beta_x) here reproduces
    `entrance.peak_beta_x`, since that is exactly what _min_peak() minimised.
    Returns (s, beta_x, beta_y, Dx, Dxp) as arrays.
    """
    lat = build_lattice(dl.geom, dl.ko, dl.ki, sliced=True, nsl=nsl)
    t0 = oc.Twiss()
    t0.E = dl.energy_gev
    t0.beta_x, t0.alpha_x = entrance.beta_x, entrance.alpha_x
    t0.beta_y, t0.alpha_y = entrance.beta_y, entrance.alpha_y
    tws = oc.twiss(lat, t0)
    return (np.array([t.s for t in tws]),
            np.array([t.beta_x for t in tws]),
            np.array([t.beta_y for t in tws]),
            np.array([t.Dx for t in tws]),
            np.array([t.Dxp for t in tws]))


# ------------------------- driver: choose the inner ki -------------------------

def design_dogleg(geom, ki_scan=None, energy_gev=0.0395, emit_ratio=1.0):
    """Scan the inner-pair ki and keep the roundest, smallest solution.

    For each ki: solve the achromat (API 1) and the min-peak entrance Twiss
    (API 2), then score with EntranceTwiss.peak_score(emit_ratio) and keep the
    smallest.

    `emit_ratio` = eps_y/eps_x of the beam that will be sent through. The
    default 1.0 scores on max(peak_beta_x, peak_beta_y) -- the original,
    distribution-free behaviour. Passing the measured ratio makes this layer
    agree with the downstream crossing match, whose roundness residual already
    weights the planes by it; leaving it at 1.0 keeps the whole dogleg design
    independent of any particle distribution.

    ki_scan defaults to linspace(-70, -6, 33) [1/m^2].
    Returns (DoglegLattice, EntranceTwiss); raises if no ki is feasible.
    """
    if ki_scan is None:
        ki_scan = np.linspace(-70.0, -6.0, 33)
    best = None
    for ki in ki_scan:
        try:
            sol = solve_achromat_quads(geom, float(ki), energy_gev=energy_gev)
            ent = solve_entrance_twiss(sol)
        except Exception:
            continue
        score = ent.peak_score(emit_ratio)
        if best is None or score < best[0]:
            best = (score, sol, ent)
    if best is None:
        raise RuntimeError("design_dogleg: no feasible ki in the scan")
    return best[1], best[2]
