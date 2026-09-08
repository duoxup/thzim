#!/usr/bin/env python3
r"""Four-dipole chicane: geometry, lattice, and the matched vertical Twiss.

The bunch compressor of the superradiant branch (OP3, OP4), and a plain drift
footprint for the SASE branch, whose dipoles are switched off. It carries NO
quadrupoles: a chicane of four RBends is achromatic by symmetry, so there is
nothing to solve for the dispersion.

Geometry is parameterised by the LONGITUDINAL footprint, because that is what is
fixed on a real beamline; theta is the tuning knob:

    rho      = L_Bz / sin(theta)                      bend radius
    L_B,arc  = rho theta = L_Bz theta / sin(theta)     dipole arc
    L_D,path = L_Dz / cos(theta)                       drift at angle theta
    Delta_x  = 2 L_Bz tan(theta/2) + L_Dz tan(theta)   mid-chicane offset

The two transverse planes behave completely differently, which is what splits
the entrance-Twiss problem into a hard solve and a soft choice:

* **x is exactly a drift.** A rectangular bend's horizontal matrix is identically
  a drift of its CHORD, rho sin(theta) = L_Bz here -- body focusing and edge
  defocusing cancel. The whole chicane is therefore [[1, L_x], [0, 1]] to machine
  precision (R21 ~ 1e-16), with L_x the sum of chords and drift paths. There is
  no equilibrium optics in x, so beta_x is a FREE choice. It is pushed to large
  values only by space charge -- bigger beam, lower density, less emittance
  growth and less SC-induced residual dispersion -- and stopped by the
  betatron size overtaking the dispersive floor eta sigma_delta. That trade-off
  lives in a tracking scan with a real distribution, NOT in this module. Note
  the beta minimising the beam SIZE (beta* ~ L_x) is the WORST for emittance.
* **y is an edge-focusing channel.** Each RBend edge focuses vertically,
  1/f = tan(theta/2)/rho, eight edges all the same sign. That channel has a
  periodic solution, matched_y() below, and injecting it keeps beta_y flat
  across the chicane -- hence no high-density waist, hence minimal SC in y. So
  the linear criterion (flat beta) and the collective one (minimal emittance
  growth) agree, and y can be solved here, exactly, without a distribution.

Nothing in this module reads a particle distribution.

The SC/CSR tracking through the chicane lives in `thzim.compressor`, shared
with the dogleg.
"""

from dataclasses import dataclass
from math import acos, degrees, radians, sin

import numpy as np
import ocelot as oc
from ocelot.cpbd.optics import lattice_transfer_map
from scipy.optimize import brentq

from thzim.utils import M_E_GEV, brho

__all__ = [
    "M_E_GEV", "brho",
    "bend_radius", "arc_length", "drift_path", "offset", "r56_z_smallangle",
    "theta_for_r56", "field",
    "ChicaneGeom", "chicane_seq", "build_lattice", "element_spans",
    "r56_z_exact", "matched_y",
]

# M_E_GEV / brho live in thzim.utils so that every module shares one definition;
# they are re-exported here (see __all__) because this module's public API has
# always carried them.


# --------------------------- geometry (vectorised) ---------------------------
# theta is in RADIANS throughout this module.

def bend_radius(theta, l_bz):
    """rho [m] of a dipole whose arc projects to L_Bz along z."""
    return l_bz / np.sin(theta)


def arc_length(theta, l_bz):
    """Dipole arc length [m]; -> L_Bz as theta -> 0."""
    return l_bz * theta / np.sin(theta)


def drift_path(theta, l_dz):
    """Path length [m] of the inter-dipole drift, travelled at angle theta."""
    return l_dz / np.cos(theta)


def offset(theta, l_bz, l_dz):
    """Transverse excursion Delta_x [m] at the middle of the chicane."""
    return 2.0 * l_bz * np.tan(0.5 * theta) + l_dz * np.tan(theta)


def r56_z_smallangle(theta, l_bz, l_dz):
    """R56 [m], z convention (> 0 for a chicane), small-angle, no velocity term.

    Reads 1.6 % low at 16.7 deg and 2.4 % low at 19.1 deg against the exact map;
    the dropped velocity term is worth only 0.2-0.6 % there. Deliberate: the
    chicane is built with margin and trimmed afterwards with theta. Use
    r56_z_exact() when the tracked value is wanted.
    """
    return 2.0 * theta**2 * (drift_path(theta, l_dz)
                             + (2.0 / 3.0) * arc_length(theta, l_bz))


def theta_for_r56(r56_target, l_bz, l_dz, bracket=(1e-3, 1.0)):
    """Bend angle [rad] giving the requested R56 at this footprint."""
    return brentq(lambda t: r56_z_smallangle(t, l_bz, l_dz) - r56_target,
                  *bracket, xtol=1e-12)


def field(theta, l_bz, energy_gev):
    """Dipole field [T] at the given total energy."""
    return brho(energy_gev) * np.sin(theta) / l_bz


# ------------------------------- the geometry -------------------------------

@dataclass
class ChicaneGeom:
    """Footprint-constrained four-RBend chicane (lengths in m, angle in deg).

    Note the contrast with `thzim.dogleg.DoglegGeom`, which is OFFSET
    constrained: there rho and Delta_x are given and the straight L_gap is
    derived. Here the LONGITUDINAL footprint is what the beamline fixes, so
    L_Bz / L_Dz / L_2 are given and rho and Delta_x are derived, with theta the
    tuning knob. That difference is the whole reason the two compressors are
    sized by different maps.
    """

    theta_deg: float
    l_bz: float                   # dipole z-projection (arc projected on z)
    l_dz: float                   # dipole-to-dipole drift z-projection
    l2: float = 0.20              # middle drift, dipole 2 -> dipole 3
    lead: float = 0.20            # entrance plane S -> B1; shared convention
    tail: float = 0.0             # bookkeeping drift B4 -> E

    @property
    def theta(self):
        return radians(self.theta_deg)

    @property
    def rho(self):
        return float(bend_radius(self.theta, self.l_bz))

    @property
    def L_bend(self):
        """Dipole arc length [m]."""
        return float(arc_length(self.theta, self.l_bz))

    @property
    def L_drift(self):
        """Inter-dipole drift path length [m], travelled at angle theta."""
        return float(drift_path(self.theta, self.l_dz))

    @property
    def delta_x(self):
        """Transverse excursion [m] at the middle of the chicane."""
        return float(offset(self.theta, self.l_bz, self.l_dz))

    @property
    def length(self):
        """Total path length S -> E [m]."""
        return (self.lead + 4.0 * self.L_bend + 2.0 * self.L_drift
                + self.l2 + self.tail)

    @property
    def L_x(self):
        """Equivalent x-drift length [m].

        Each RBend contributes only its CHORD rho sin(theta) = L_Bz in the
        horizontal plane, not its arc, so the whole chicane is exactly
        [[1, L_x], [0, 1]] in x. Shorter than `length` by 4 (L_bend - L_Bz).
        """
        return (self.lead + 4.0 * self.l_bz + 2.0 * self.L_drift
                + self.l2 + self.tail)

    @property
    def z_footprint(self):
        """Longitudinal extent [m]: what the SASE branch replaces with a drift
        when the dipoles are switched off."""
        return (self.lead + 4.0 * self.l_bz + 2.0 * self.l_dz
                + self.l2 + self.tail)

    @property
    def R56_z(self):
        """Small-angle R56 [m], z convention, no velocity term."""
        return float(r56_z_smallangle(self.theta, self.l_bz, self.l_dz))

    def field(self, energy_gev):
        """Dipole field [T] at the given total energy."""
        return float(field(self.theta, self.l_bz, energy_gev))

    def summary(self):
        return (
            f"Chicane  (theta={self.theta_deg:.3f} deg, L_Bz={self.l_bz:.3f}, "
            f"L_Dz={self.l_dz:.3f}, L_2={self.l2:.3f} m)\n"
            f"  derived: rho={self.rho:.4f} m  L_bend={self.L_bend:.4f}  "
            f"L_drift={self.L_drift:.4f}  Delta_x={self.delta_x*1e3:.1f} mm\n"
            f"  lengths: path={self.length:.4f}  x-drift={self.L_x:.4f}  "
            f"z-footprint={self.z_footprint:.4f} m  "
            f"(lead={self.lead}, tail={self.tail})\n"
            f"  R56_z={self.R56_z*1e3:+.2f} mm (small-angle, no velocity term)")


# -------------------------------- the lattice --------------------------------

def chicane_seq(geom):
    """Four-RBend chicane (+t, -t, -t, +t) with S / C (centre) / E markers.

    RBend, not SBend: sector bends do not close the dispersion here.
    Returns (sequence, centre_marker).
    """
    l_b, l_d = geom.L_bend, geom.L_drift
    mc = oc.Marker(eid="C")
    seq = [oc.Marker(eid="S")]
    if geom.lead > 0:
        seq += [oc.Drift(l=geom.lead)]
    seq += [oc.RBend(l=l_b, angle=+geom.theta), oc.Drift(l=l_d),
            oc.RBend(l=l_b, angle=-geom.theta), oc.Drift(l=geom.l2 / 2), mc,
            oc.Drift(l=geom.l2 / 2), oc.RBend(l=l_b, angle=-geom.theta),
            oc.Drift(l=l_d), oc.RBend(l=l_b, angle=+geom.theta)]
    if geom.tail > 0:
        seq += [oc.Drift(l=geom.tail)]
    return seq + [oc.Marker(eid="E")], mc


def build_lattice(geom):
    """The chicane as an Ocelot MagneticLattice."""
    return oc.MagneticLattice(chicane_seq(geom)[0])


def element_spans(geom):
    """(s_start, s_end, kind) of every dipole, s measured from the S marker.

    kind is "bend" throughout -- a chicane has no quadrupoles. Layout knowledge,
    so it lives with the geometry rather than in whichever script draws it.
    """
    l_b, l_d = geom.L_bend, geom.L_drift
    spans, s = [], geom.lead
    for length, is_bend in ((l_b, True), (l_d, False), (l_b, True),
                            (geom.l2, False), (l_b, True), (l_d, False),
                            (l_b, True)):
        if is_bend:
            spans.append((s, s + length, "bend"))
        s += length
    return spans


def r56_z_exact(geom, energy_gev):
    """R56 [m] in the z convention (> 0 for a chicane), from the Ocelot map.

    Includes the velocity term, so it depends on energy and on how much drift
    the lead/tail bookkeeping puts inside the lattice.
    """
    lat = build_lattice(geom)
    lattice_transfer_map(lat, energy_gev)
    return -lat.R[4, 5]


# ------------------------ the matched vertical solution ------------------------

def matched_y(geom, energy_gev):
    """Periodic (matched) y-Twiss of the half chicane, S -> C.

    The eight RBend edges make a vertical focusing channel; its periodic
    solution is the entrance Twiss that keeps beta_y flat across the whole
    chicane instead of forming a high-density waist. From the half-chicane
    y-matrix,

        cos(mu) = (m11 + m22)/2,   beta* = m12/sin(mu),
        alpha*  = (m11 - m22)/(2 sin(mu))

    Returns (beta [m], alpha, mu [deg]), or None if the channel is unstable
    (|m11 + m22| >= 2), which is a genuine upper bound on theta: mu -> 180 deg
    is a half-integer resonance of the half cell and beta* diverges there. At
    L_Bz = 0.1 / L_Dz = 0.75 / L_2 = 0.2 that limit sits between 27 and 28 deg,
    far above the OP3/OP4 working angles.

    `geom.lead` is NOT cosmetic. It puts a drift inside the cell whose periodic
    solution is being taken, so it shifts the answer: at OP3 (theta = 16.69 deg,
    L_Dz = 0.75) beta* runs 0.9457 -> 0.9568 -> 0.9931 m for lead = 0 -> 0.2 ->
    0.5 m, and alpha* changes SIGN, -0.0530 -> +0.0524 -> +0.2103. `lead = 0`
    solves the magnetic channel alone, an arbitrary-free quantity; the real
    interface-plane-to-first-bend distance gives the target at that plane
    instead. The project convention is `lead = 0.2 m`, and that is the
    `ChicaneGeom` default.

    What the choice does NOT change is the flatness that motivates the whole
    exercise: beta_y stays within [0.65, 1.06] m across that whole range of
    lead, against [0.023, 59] m for a mismatched injection.

    Note also that exit self-reproduction needs lead == tail; the mirror
    symmetry is broken otherwise.
    """
    seq, mc = chicane_seq(geom)
    sub = oc.MagneticLattice(seq[:seq.index(mc) + 1])
    lattice_transfer_map(sub, energy_gev)
    m11, m12, m22 = sub.R[2, 2], sub.R[2, 3], sub.R[3, 3]
    trace = m11 + m22
    if abs(trace) >= 2.0:
        return None
    mu = acos(trace / 2.0)
    s = sin(mu) if m12 >= 0 else -sin(mu)
    if s == 0:
        return None
    return m12 / s, (m11 - m22) / (2 * s), degrees(mu if m12 >= 0 else -mu)
