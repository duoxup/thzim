r"""The design of record: every working point's middle-optics line, as data.

One place that says, per operating point, what the line between the booster
exit (P0) and the undulator entrance (P3) IS -- the geometry of each section
and the strengths selected for it -- so that a driver can rebuild it without
re-running any design step. The values are those the per-OP scripts under
`repro/` derived and printed; the scripts remain the place where
each number is DERIVED (and the place to change one), this module is where the
result is RECORDED. Where a value follows from another by a deterministic
solve (the dogleg's outer-pair `ko` from `ki`), it is solved here, not copied.

The line has three sections, joined at the interface planes:

    P0 --[M1]--> P1 --[compressor]--> P2 --[M3]--> P3

* SASE branch (OP1, OP2): M1 is the two-triplet crossing match over the
  switched-off chicane footprint (`thzim.two_triplet`), the compressor is the
  achromatic dogleg (`thzim.dogleg`), M3 is the four-quad station after it
  (`thzim.quadruplet`).
* Superradiant branch (OP3, OP4): M1 is the four-quad station
  (`thzim.quadruplet`), the compressor is the chicane (`thzim.chicane`), M3 is
  T1.Q3 plus the T2 triplet across the switch region, run as a quadruplet with
  T1's first two quads switched off (`thzim.quadruplet` with per-gap spacing).

Each section also records the SPACE-CHARGE TRACKING SETTINGS it was designed
under (navigator step, SC mesh), because they are part of the result: the
two-triplet crossings were selected on 31^3 / 0.05 m tracks, the quadruplet
re-matches and the compressor runs on 63^3 / 0.02 m, and re-tracking OP2's
two-triplet section at 63^3 / 0.02 m moved its P2 peak current from 213 to
335 A (MIGRATION.md) -- the compression sits near
its knee and feels the transverse conditions through space charge. A driver
that wants to reproduce the design must use each section's own settings.

Strength conventions follow the module that solved them: the two-triplet
sections carry GRADIENTS [T/m], and the k1 the lattice is built with comes
from the energy of the BEAM being tracked, exactly as `two_triplet_e2e.py`
does (`match_with` reads it off the distribution) -- not from the nominal
`ekin_mev`. The two differ by 0.3 % (the P0 beams sit at 15.95 / 40.05 MeV
total against nominal 15.91 / 39.91), and OP2's M1 is focused hard enough
that converting with the nominal energy moves its P1 beta_x from 1.78 to
1.41 m. The quadruplets carry k1 [1/m^2], which is what their solver returns
and is energy-free. The quadruplet k1 are the
SPACE-CHARGE-matched values (the `sc_rematch` result), since that is the
lattice the design tracks; the linear seeds are in the scripts.

The P3 targets are FEL-side choices with no derivation script; they are
recorded here as the numbers the M3 matches were solved against.
"""

from dataclasses import dataclass

from thzim import chicane, dogleg, quadruplet, two_triplet
from thzim.chicane import ChicaneGeom
from thzim.dogleg import DoglegGeom
from thzim.quadruplet import QuadrupletGeom
from thzim.two_triplet import TwoTripletGeom
from thzim.utils import M_E_GEV

__all__ = [
    "PLANES", "QuadrupletKnobs", "TwoTripletKnobs", "DoglegKnobs",
    "ChicaneKnobs", "Segment", "OpRecord", "RECORDS", "get",
]

PLANES = ("p0", "p1", "p2", "p3")


# --------------------------------- knobs ---------------------------------

@dataclass(frozen=True)
class QuadrupletKnobs:
    """A four-quad section with its k1 [1/m^2] in beam order."""
    geom: QuadrupletGeom
    k1s: tuple
    unit_step: float = 0.02       # the settings of m1_/m3_quadruplet_match.py
    sc_mesh: tuple = (63, 63, 63)

    def lattice(self, energy_gev):
        return quadruplet.build_lattice(self.geom, list(self.k1s))

    def spans(self):
        return self.geom.element_spans()

    def describe(self):
        return ("quadruplet k1 = (" + ", ".join(f"{k:+.3f}" for k in self.k1s)
                + ") 1/m^2")


@dataclass(frozen=True)
class TwoTripletKnobs:
    """Two triplets around a gap, gradients [T/m] in PHYSICAL order."""
    geom: TwoTripletGeom
    t1_g: tuple
    t2_g: tuple
    unit_step: float = 0.05       # the settings of two_triplet_scan/e2e.py
    sc_mesh: tuple = (31, 31, 31)

    def lattice(self, energy_gev):
        return two_triplet.build_lattice(self.geom, self.t1_g, self.t2_g,
                                         energy_gev)

    def spans(self):
        return self.geom.element_spans()

    def describe(self):
        return ("two-triplet T1 = (" + ", ".join(f"{g:+.4f}" for g in self.t1_g)
                + "), T2 = (" + ", ".join(f"{g:+.4f}" for g in self.t2_g)
                + ") T/m")


@dataclass(frozen=True)
class DoglegKnobs:
    """The achromatic dogleg: geometry plus the free inner-pair `ki`;
    the outer-pair `ko` is solved from the achromat condition on demand."""
    geom: DoglegGeom
    ki: float
    unit_step: float = 0.02       # the settings of dogleg_track.py
    sc_mesh: tuple = (63, 63, 63)

    def lattice(self, energy_gev):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):     # ocelot's matcher
            return dogleg.solve_achromat_quads(self.geom, self.ki,
                                               energy_gev=energy_gev).lattice

    def spans(self):
        return dogleg.element_spans(self.geom)

    def describe(self):
        g = self.geom
        return (f"dogleg theta={g.theta_deg:g} deg rho={g.rho:g} m "
                f"Delta_x={g.delta_x:g} m d_in={g.d_in:g} d_out={g.d_out:g}, "
                f"ki={self.ki:+g} 1/m^2 (ko from the achromat)")


@dataclass(frozen=True)
class ChicaneKnobs:
    """The chicane: geometry only, it has no quadrupoles."""
    geom: ChicaneGeom
    unit_step: float = 0.02       # the settings of chicane_track.py
    sc_mesh: tuple = (63, 63, 63)

    def lattice(self, energy_gev):
        return chicane.build_lattice(self.geom)

    def spans(self):
        return chicane.element_spans(self.geom)

    def describe(self):
        g = self.geom
        return (f"chicane theta={g.theta_deg:g} deg L_Bz={g.l_bz:g} "
                f"L_Dz={g.l_dz:g} lead={g.lead:g} m")


# -------------------------------- the line --------------------------------

@dataclass(frozen=True)
class Segment:
    """One section of the line, between two interface planes."""
    name: str            # "M1", "compressor", "M3"
    plane_in: str        # "p0", "p1", "p2"
    plane_out: str       # "p1", "p2", "p3"
    knobs: object        # one of the *Knobs above
    csr: bool            # does the design track this section with CSR

    def lattice(self, energy_gev):
        return self.knobs.lattice(energy_gev)

    def spans(self):
        """(s_start, s_end, kind) of the magnets, s from the segment start."""
        return self.knobs.spans()

    @property
    def unit_step(self):
        """Navigator step [m] the section was designed with."""
        return self.knobs.unit_step

    @property
    def sc_mesh(self):
        """SC mesh (nx, ny, nz) the section was designed with."""
        return tuple(self.knobs.sc_mesh)


@dataclass(frozen=True)
class OpRecord:
    """One working point's line, P0 -> P3."""
    name: str
    branch: str              # "SASE" or "superradiant"
    frequency_thz: float
    charge_nc: float
    ekin_mev: float          # nominal kinetic energy (labels; the beam's own
                             # energy converts gradients, see the module doc)
    m1: object
    compressor: object
    m3: object
    p3_target: tuple         # (beta_x, alpha_x, beta_y, alpha_y) at P3

    @property
    def energy_gev(self):
        """Nominal TOTAL energy [GeV]."""
        return (self.ekin_mev + M_E_GEV * 1e3) * 1e-3

    @property
    def segments(self):
        return (Segment("M1", "p0", "p1", self.m1, csr=False),
                Segment("compressor", "p1", "p2", self.compressor, csr=True),
                Segment("M3", "p2", "p3", self.m3, csr=False))

    def summary(self):
        lines = [f"{self.name}: {self.branch} branch, {self.frequency_thz:g} THz, "
                 f"{self.charge_nc:g} nC, E_kin {self.ekin_mev:g} MeV"]
        for seg in self.segments:
            lines.append(f"  {seg.name:<10s} {seg.plane_in} -> {seg.plane_out}"
                         f"{'  (SC + CSR' if seg.csr else '  (SC'}, "
                         f"{'x'.join(map(str, seg.sc_mesh))}, "
                         f"step {seg.unit_step:g} m): {seg.knobs.describe()}")
        t = self.p3_target
        lines.append(f"  P3 target  bx={t[0]:g} ax={t[1]:g}  by={t[2]:g} ay={t[3]:g}")
        return "\n".join(lines)


# ----------------------------- the four records -----------------------------
# Sources: the SC-matched results printed by the repro scripts,
# runs of 2026-09-08. Geometry is repeated from each script's settings block.

_SASE_M1_GEOM = TwoTripletGeom(t1_lq=(0.10, 0.10, 0.10),
                               t1_drifts=(0.30, 0.30, 0.30),
                               t2_lq=(0.10, 0.10, 0.10),
                               t2_drifts=(0.30, 0.30, 0.40),
                               gap=3.40, screens=(0.34, 1.70, 3.06))

# T1.Q3 + T2 across the 0.92 m switch region; T1.Q1, T1.Q2 switched off
_SR_M3_GEOM = QuadrupletGeom(lq=0.10, d_in=0.90, d_inter=(0.92, 0.30, 0.30),
                             d_out=0.40)

RECORDS = {
    "OP1": OpRecord(
        name="OP1", branch="SASE", frequency_thz=1.0, charge_nc=1.0,
        ekin_mev=15.4,
        m1=TwoTripletKnobs(_SASE_M1_GEOM,                 # two_triplet_e2e.py
                           t1_g=(-0.2647, +0.5131, -0.2649),
                           t2_g=(-0.6401, +0.7404, +0.2494)),
        compressor=DoglegKnobs(DoglegGeom(theta_deg=40.0, rho=0.542,
                                          delta_x=2.0, d_in=0.2, d_out=0.1),
                               ki=-38.0),                 # dogleg_track.py
        m3=QuadrupletKnobs(QuadrupletGeom(lq=0.10, d_in=0.10, d_inter=0.30,
                                          d_out=0.40),    # m3_quadruplet_match.py
                           k1s=(+35.405, -57.661, +34.646, -46.402)),
        p3_target=(3.38, 0.0, 0.007, 0.0)),
    "OP2": OpRecord(
        name="OP2", branch="SASE", frequency_thz=10.0, charge_nc=1.0,
        ekin_mev=39.4,
        m1=TwoTripletKnobs(_SASE_M1_GEOM,
                           t1_g=(-3.2756, +2.4942, -1.0506),
                           t2_g=(-3.3110, +4.3513, -5.5767)),
        compressor=DoglegKnobs(DoglegGeom(theta_deg=40.0, rho=0.542,
                                          delta_x=2.0, d_in=0.0, d_out=0.3),
                               ki=-22.0),
        m3=QuadrupletKnobs(QuadrupletGeom(lq=0.10, d_in=0.30, d_inter=0.30,
                                          d_out=0.40),
                           k1s=(+23.285, -34.104, +33.890, -126.630)),
        p3_target=(0.53, 0.0, 0.53, 0.0)),
    "OP3": OpRecord(
        name="OP3", branch="superradiant", frequency_thz=1.0, charge_nc=0.2,
        ekin_mev=39.9,
        m1=QuadrupletKnobs(QuadrupletGeom(),                # m1_quadruplet_match.py
                           k1s=(-14.706, +37.151, -41.200, +22.793)),
        compressor=ChicaneKnobs(ChicaneGeom(theta_deg=16.69, l_bz=0.10,
                                            l_dz=0.75, lead=0.20)),
        m3=QuadrupletKnobs(_SR_M3_GEOM,
                           k1s=(+14.140, -22.705, +32.765, -23.183)),
        p3_target=(2.58, 0.0, 0.07, 0.0)),
    "OP4": OpRecord(
        name="OP4", branch="superradiant", frequency_thz=0.3, charge_nc=0.6,
        ekin_mev=21.8,
        m1=QuadrupletKnobs(QuadrupletGeom(),
                           k1s=(+11.074, -27.492, +45.430, +98.143)),
        compressor=ChicaneKnobs(ChicaneGeom(theta_deg=19.05, l_bz=0.10,
                                            l_dz=0.75, lead=0.20)),
        m3=QuadrupletKnobs(_SR_M3_GEOM,
                           k1s=(+23.780, -15.578, +31.544, -36.485)),
        p3_target=(9.8, 0.0, 0.006, 0.0)),
}


def get(name):
    """The record for "OP1".."OP4" (case-insensitive); KeyError otherwise."""
    key = name.upper()
    if key not in RECORDS:
        raise KeyError(f"unknown operating point {name!r}; "
                       f"known: {', '.join(RECORDS)}")
    return RECORDS[key]
