r"""Compressor tracking: a bunch through a dogleg or a chicane with SC and CSR.

The one stage of the middle optics that is NOT a matching problem. The lattice
is fixed by the design of record (`thzim.dogleg` for the SASE branch,
`thzim.chicane` for the superradiant branch) and the question is only what the
beam does in it: how far it compresses, what the collective effects cost, and
what leaves at P2 for the final match. So this module solves nothing; it
tracks, records, and summarises. Both compressors share it because the
questions are the same and only the lattice differs.

What is recorded along s, and why each quantity is there:

* `sig_x` and `sig_xb` -- the rms size and its BETATRON part, with the linear
  dispersive term `eta sigma_delta` removed by regression on delta. Inside a
  compressor the dispersive part dominates (eta reaches 0.25 m in the chicane
  against a betatron size of a fraction of a mm), so the raw size says little
  about the optics; the betatron size is what the quads and the space charge
  actually act on.
* `eta` -- the STATISTICAL dispersion <x delta>/<delta^2>, which tracks the
  linear eta through the bends and, at the exit, is the residual left by
  collective effects (the linear lattice closes it to ~1e-4 m).
* `enx_c`, `eny` -- the dispersion-corrected horizontal and the projected
  vertical normalised emittance. The PROJECTED horizontal emittance is recorded
  too (`enx`) but is bookkeeping inside the bends: eta takes it to ~100 um and
  back, which is not growth. Only its exit value, where the two agree, means
  anything.
* `sig_z`, `I_peak`, `sig_dp`, `mean_dp` -- the longitudinal story: the bunch
  length and peak current through the compression, and the energy spread and
  the mean energy shift (which is the CSR loss, since space charge conserves
  the beam's energy).

The linear reference is the same track with `sc=False, csr=False`; the monitor
is a physics process itself, so the sampling density is `unit_step` either way.

Space charge in bends: ocelot's `SpaceCharge` is a straight-line solver applied
element by element, which is the accepted approximation at these energies; CSR
is ocelot's 1-D projected model (`CSR`, `n_bin` longitudinal bins). Neither is
changed here. The conversion from partdist goes through `thzim.triplet.
as_ocelot`, which centres `tau` -- see its docstring for why that matters to
the field solvers.

Migrated from the tracking half of the earlier chicane study implementation
(`track_chicane`, `Monitor`); generalised to take any lattice so the dogleg
uses it too. See MIGRATION.md.
"""

import contextlib
import io

import numpy as np
import ocelot as oc
from ocelot.cpbd.csr import CSR
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.physics_proc import PhysProc
from ocelot.cpbd.sc import SpaceCharge
from ocelot.cpbd.track import track
from scipy.constants import c as C_LIGHT

from thzim.triplet import as_ocelot
from thzim.utils import M_E_GEV

__all__ = [
    "emit_proj", "emit_corr", "dispersion", "peak_current", "chirp",
    "BeamMonitor", "track_compressor", "beam_report",
]


# ------------------------------ beam statistics ------------------------------

def emit_proj(u, up):
    """Geometric rms emittance [m] of one plane, projected."""
    u = u - u.mean()
    up = up - up.mean()
    return float(np.sqrt(max(u.var() * up.var() - np.mean(u * up) ** 2, 0.0)))


def dispersion(u, d):
    """Statistical dispersion <u delta>/<delta^2> [m or rad]; 0 if delta is flat."""
    d = d - d.mean()
    vd = d.var()
    return float(np.mean((u - u.mean()) * d) / vd) if vd > 0 else 0.0


def emit_corr(u, up, d):
    """Geometric rms emittance [m] with the linear dispersive part removed.

    Regresses u and u' on delta and takes the emittance of the residual, so the
    result is the BETATRON emittance where the beam is dispersive and equals
    `emit_proj` where it is not.
    """
    d = d - d.mean()
    ub = (u - u.mean()) - dispersion(u, d) * d
    upb = (up - up.mean()) - dispersion(up, d) * d
    return emit_proj(ub, upb)


def peak_current(tau, q, nb=300, smooth=5):
    """Peak current [A] from a charge-weighted histogram of tau.

    Bins between the 1st and 99th percentile, so a few stray particles cannot
    stretch the axis, and smooths over `smooth` bins to keep the peak from
    being a single noisy bin. `q` is the per-particle charge magnitude [C].
    """
    lo, hi = np.percentile(tau, [1.0, 99.0])
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, nb + 1)
    h, _ = np.histogram(tau, bins=edges, weights=q)
    ii = h * C_LIGHT / (edges[1] - edges[0])
    if smooth > 1:
        ii = np.convolve(ii, np.ones(smooth) / smooth, mode="same")
    return float(ii.max())


def chirp(tau, dp, core=2.0):
    """Linear energy chirp d(delta)/d(tau) [1/m] of the core (|tau| < core sigma).

    In ocelot's convention `tau = s - z` is positive for particles BEHIND the
    reference, so the sign is opposite to a chirp quoted against z.
    """
    t = tau - tau.mean()
    m = np.abs(t) < core * t.std()
    return float(np.polyfit(t[m], dp[m], 1)[0])


# --------------------------------- monitor ---------------------------------

class BeamMonitor(PhysProc):
    """Records the beam moments at every navigator step.

    A physics process with no effect on the beam; it exists so the record
    lands at `unit_step` rather than at element boundaries. Records are plain
    dicts, collected into arrays by `evolution()`.
    """

    def __init__(self):
        super().__init__()
        self.rec = []

    def apply(self, p, dz):
        x, xp, y, yp, tau, dp = p.rparticles
        q = np.abs(p.q_array)
        gamma = p.E / M_E_GEV
        d = dp - dp.mean()
        eta, etap = dispersion(x, d), dispersion(xp, d)
        xb = (x - x.mean()) - eta * d
        eps_c = emit_corr(x, xp, dp)
        eps_y = emit_proj(y, yp)
        self.rec.append(dict(
            s=float(p.s), E=float(p.E), mean_dp=float(dp.mean()),
            sig_x=float(x.std()), sig_xb=float(xb.std()), sig_y=float(y.std()),
            sig_z=float(tau.std()), sig_dp=float(d.std()),
            eta=eta, etap=etap,
            beta_x=float(xb.var() / eps_c) if eps_c > 0 else np.nan,
            beta_y=float((y - y.mean()).var() / eps_y) if eps_y > 0 else np.nan,
            enx=gamma * emit_proj(x, xp), enx_c=gamma * eps_c, eny=gamma * eps_y,
            I_peak=peak_current(tau, q)))

    def evolution(self, s0=None):
        """The record as a dict of arrays; `s` is measured from `s0`
        (default: the first record)."""
        evo = {k: np.array([r[k] for r in self.rec]) for k in self.rec[0]}
        evo["s"] = evo["s"] - (evo["s"][0] if s0 is None else s0)
        return evo


# --------------------------------- tracking ---------------------------------

def track_compressor(lattice, dist, sc=True, csr=True, unit_step=0.02,
                     nmesh=(63, 63, 63), csr_nbin=300):
    """Track a partdist distribution through `lattice` with SC and/or CSR.

    Returns ``(evo, pa_out)``: the `BeamMonitor.evolution()` record with `s`
    measured from the lattice entrance, and the exit ocelot ParticleArray
    (convert back with `partdist.from_ocelot_particle_array`). `sc=False,
    csr=False` is the linear reference, sampled just as densely.

    `unit_step` is both the collective-kick step and the record spacing. The
    CSR model needs its own sub-steps and sets them itself.
    """
    seq = lattice.sequence
    navi = Navigator(lattice)
    navi.unit_step = unit_step
    if csr:
        proc = CSR()
        proc.n_bin = csr_nbin
        navi.add_physics_proc(proc, seq[0], seq[-1])
    if sc:
        proc = SpaceCharge()
        proc.nmesh_xyz = list(nmesh)
        proc.step = 1
        navi.add_physics_proc(proc, seq[0], seq[-1])
    mon = BeamMonitor()
    navi.add_physics_proc(mon, seq[0], seq[-1])
    pa = as_ocelot(dist)
    with contextlib.redirect_stdout(io.StringIO()):       # CSR is chatty
        track(lattice, pa, navi, print_progress=False, calc_tws=False)
    mon.apply(pa, 0.0)           # the navigator's last step stops short of the
    return mon.evolution(), pa   # exit marker; record the exit plane explicitly


def beam_report(pa):
    """Scalar summary of an ocelot ParticleArray, the exit-plane counterpart of
    `BeamMonitor.apply` plus the Twiss and the chirp."""
    x, xp, y, yp, tau, dp = pa.rparticles
    q = np.abs(pa.q_array)
    gamma = pa.E / M_E_GEV
    d = dp - dp.mean()
    eta, etap = dispersion(x, d), dispersion(xp, d)
    xb = (x - x.mean()) - eta * d
    xpb = (xp - xp.mean()) - etap * d
    eps_c, eps_y = emit_proj(xb, xpb), emit_proj(y, yp)
    yc, ypc = y - y.mean(), yp - yp.mean()
    return dict(
        E_mev=pa.E * 1e3, mean_dp=float(dp.mean()), q_nc=q.sum() * 1e9,
        sig_x=float(x.std()), sig_xb=float(xb.std()), sig_y=float(y.std()),
        sig_z=float(tau.std()), sig_dp=float(d.std()), eta=eta, etap=etap,
        enx=gamma * emit_proj(x, xp), enx_c=gamma * eps_c, eny=gamma * eps_y,
        I_peak=peak_current(tau, q), chirp=chirp(tau, dp),
        beta_x=float(xb.var() / eps_c), alpha_x=float(-np.mean(xb * xpb) / eps_c),
        beta_y=float(yc.var() / eps_y), alpha_y=float(-np.mean(yc * ypc) / eps_y))
