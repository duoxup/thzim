#!/usr/bin/env python3
r"""Magnetic bunch-compressor parameter scan (linear theory).

Answers the sizing-stage question "how much R56 do I need, and what chirp does
that demand from the injector?" without touching a real distribution: the whole
map is analytic, so the compressor footprint can be fixed before any tracking.

Grid axes: rms bunch length sigma_z,b BEFORE compression  x  compressor R56.
Colour map: the correlated momentum spread the injector must deliver.

``__main__`` draws one panel per operating point (2x2). Each panel's background
is built from THAT point's own charge and target current, and the beam it
describes is overlaid twice: as the iso-line at the spread the injector
delivers, and as a marker at the (sigma_z,b, R56) that spread implies. Marker
and line coincide by construction -- if they ever drift apart, a panel is being
drawn with the wrong charge.

Every beam input is a free parameter in the settings block -- charge, correlated
spread, bunch length before compression, target current. They are seeded with
the values measured from data/OP*_50k.dist, but nothing in the main flow reads a
distribution, so the tool answers "what if" questions as readily as "what is".
To refresh the seeds after the beams change, call measure_beam() (the only
function needing partdist, imported lazily):

    python -c "from bunch_compression_scan import measure_beam; \
               print(measure_beam('OP1', '../data/OP1_50k.dist'))"

Longitudinal map (linear, two sign conventions in circulation):

    z convention:    z1 = z0 + R56 delta,   delta = h_z   z0   ->  M = 1 + h_z R56
    tau convention:  t1 = t0 - R56 delta,   delta = h_tau t0   ->  M = 1 - h_tau R56

with the compression factor C = sigma_initial / sigma_final = 1/|M|, so

    under-compression:  M = +1/C          over-compression:  M = -1/C

and the required correlated spread is sigma = |h| sigma_z,b0.

WHAT THE COLOUR MAP ACTUALLY IS
-------------------------------
delta is the relative MOMENTUM deviation dp/p0 -- the quantity conjugate to R56
-- which in the paraxial limit equals dp_z/p_z0 (they differ by theta^2/2, i.e.
the divergence squared: < 1e-7 for these beams). It is NOT the relative energy
spread: dE/E0 = beta^2 dp/p0, so

    sigma_E/E0 = (1 - 1/gamma^2) sigma_delta

which is a 1e-4..1e-3 correction at 15-40 MeV. Earlier versions of this script
plotted dp_z/p_z0 while labelling it "energy spread"; the label now says what
the maths computes, and setting ekin_ev prints the energy-spread conversion.

Also note this is the CORRELATED (chirp) spread only. The slice/uncorrelated
spread that sets the FEL bandwidth is a separate, much smaller quantity (a few
percent of the total at P0) and does not enter this model.

Layout: profile models -> compression physics -> scan_compression()
-> ``__main__`` (user settings, compute, plot). Figures are shown, not saved.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c_light, e as q_e, m_e
import plt_style as ps
from pathlib import Path

FWHM_OVER_SIGMA_GAUSS = 2.0 * np.sqrt(2.0 * np.log(2.0))   # 2.3548
FWHM_OVER_SIGMA_PARABOLA = np.sqrt(10.0)                   # 3.1623


# ------------------------ longitudinal profile models ------------------------

@dataclass(frozen=True)
class Profile:
    """A longitudinal current profile, used only to link I_peak to sigma_z."""

    label: str
    sigma_z_at: Callable[[float, float], float]   # (charge, peak_current) -> sigma_z
    fwhm_over_sigma: float


def sigma_z_gaussian(charge, peak_current):
    """Gaussian bunch: I_peak = Q c / (sqrt(2 pi) sigma_z)."""
    return charge * c_light / (np.sqrt(2.0 * np.pi) * peak_current)


def sigma_z_parabola(charge, peak_current):
    """Inverted parabola I(t) = I_peak (1 - (t/tau)^2):
    Q = 4 tau I_peak / 3 and sigma_t = tau / sqrt(5)."""
    return 3.0 * charge * c_light / (4.0 * np.sqrt(5.0) * peak_current)


PROFILES = {
    "gaussian": Profile("Gaussian", sigma_z_gaussian, FWHM_OVER_SIGMA_GAUSS),
    "parabola": Profile("Inverted parabola", sigma_z_parabola,
                        FWHM_OVER_SIGMA_PARABOLA),
}


def peak_current(charge, sigma_z, profile):
    """I_peak [A] from charge and rms length -- the inverse of profile.sigma_z_at.

    sigma_z_at(Q, I) = k Q c / I, so sigma_z_at(Q, 1) = k Q c and the inverse is
    just that constant over sigma_z. Deriving it from the profile keeps the two
    directions consistent whatever shape is selected.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return profile.sigma_z_at(charge, 1.0) / np.asarray(sigma_z)


# ---------------------------- compression physics ----------------------------

def map_factor(compression, branch):
    """Longitudinal map element M = sigma_final / sigma_initial (signed)."""
    if branch == "under":
        return 1.0 / compression
    if branch == "over":
        return -1.0 / compression
    raise ValueError(f"branch must be 'under' or 'over', got {branch!r}")


def required_chirp(compression, r56, convention, branch):
    """Chirp h [1/m] that realises the requested compression."""
    m = map_factor(compression, branch)
    if convention == "z":
        return (m - 1.0) / r56
    if convention == "tau":
        return (1.0 - m) / r56
    raise ValueError(f"convention must be 'z' or 'tau', got {convention!r}")


def beta_squared(ekin_ev):
    """beta^2 = 1 - 1/gamma^2, the sigma_delta -> sigma_E/E0 conversion factor."""
    gamma = 1.0 + ekin_ev * q_e / (m_e * c_light**2)
    return 1.0 - 1.0 / gamma**2


def required_r56(sigma_z0, spread, sigma_z_target, convention="z", branch="under"):
    """R56 [m] compressing sigma_z0 to sigma_z_target with the spread available.

    Inverts the same map as required_chirp(); `spread` must be signed in the
    chosen convention, which is what fixes the sign of R56 (and hence whether
    the compressor is a dogleg or a chicane).
    """
    m = map_factor(sigma_z0 / sigma_z_target, branch)
    if convention == "z":
        return (m - 1.0) * sigma_z0 / spread
    if convention == "tau":
        return (1.0 - m) * sigma_z0 / spread
    raise ValueError(f"convention must be 'z' or 'tau', got {convention!r}")


# --------------------------- measured beam input ---------------------------

@dataclass
class BeamPoint:
    """What an actual injector beam offers, as read from an ASTRA dump."""

    name: str
    charge: float         # bunch charge [C]
    sigma_z: float        # rms bunch length [m]
    pz0: float            # mean longitudinal momentum [eV/c]
    spread_total: float   # std(pz)/pz0, unsigned
    spread_cor: float     # cor_pz/pz0, SIGNED in the z convention
    i_peak: float         # peak current as delivered [A]


def measure_beam(name, path):
    """Read an ASTRA distribution and pull out the compression-relevant moments.

    `spread_cor` uses partdist's cor_pz = cov(z, pz)/std(z), i.e. the CORRELATED
    part and, unlike std(pz), signed -- the sign is what places the working point
    on the correct side of R56 = 0. It sits within ~0.2 % of std(pz)/pz0 here
    because the chirp dominates the total spread at P0.
    """
    from partdist import read_astra_distribution

    d = read_astra_distribution(path)
    pz0 = d.mean("pz")
    return BeamPoint(name=name, charge=abs(d.get_data("Q").sum()),
                     sigma_z=d.std("z"), pz0=pz0,
                     spread_total=d.std("pz") / pz0, spread_cor=d.cor_pz / pz0,
                     i_peak=d.I_peak)


# ------------------------------ the scan itself ------------------------------

@dataclass
class CompressionScan:
    """Required chirp and momentum spread over a (sigma_z,b0, R56) grid."""

    sigma_z0: np.ndarray     # initial rms bunch length [m], 2-D grid
    r56: np.ndarray          # compressor R56 [m], 2-D grid
    compression: np.ndarray  # C = sigma_initial / sigma_final
    chirp: np.ndarray        # required h [1/m], signed per convention
    spread: np.ndarray       # required correlated dp/p0, signed per convention
    sigma_z_target: float    # compressed rms bunch length [m]
    profile: Profile
    convention: str
    branch: str

    @property
    def chirp_symbol(self):
        return r"h_z" if self.convention == "z" else r"h_\tau"

    def summary(self):
        return (f"{self.profile.label}, {self.branch}-compression, "
                f"{self.convention} convention\n"
                f"  target sigma_z,b = {self.sigma_z_target*1e3:.4f} mm "
                f"(FWHM = {self.sigma_z_target*self.profile.fwhm_over_sigma/c_light*1e12:.2f} ps)\n"
                f"  C in [{self.compression.min():.2f}, {self.compression.max():.2f}]  "
                f"required |dp/p0| in [{np.abs(self.spread).min()*100:.2f}, "
                f"{np.abs(self.spread).max()*100:.2f}] %")


def scan_compression(sigma_z0_axis, r56_axis, charge, peak_current,
                     distribution="gaussian", convention="z", branch="under"):
    """Evaluate the required chirp and correlated momentum spread over a grid."""
    if distribution not in PROFILES:
        raise ValueError(f"distribution must be one of {sorted(PROFILES)}, "
                         f"got {distribution!r}")
    profile = PROFILES[distribution]
    sigma_z_target = profile.sigma_z_at(charge, peak_current)

    sigma_z0, r56 = np.meshgrid(np.asarray(sigma_z0_axis), np.asarray(r56_axis))
    compression = sigma_z0 / sigma_z_target
    chirp = required_chirp(compression, r56, convention, branch)
    return CompressionScan(sigma_z0=sigma_z0, r56=r56, compression=compression,
                           chirp=chirp, spread=chirp * sigma_z0,
                           sigma_z_target=sigma_z_target, profile=profile,
                           convention=convention, branch=branch)


if __name__ == "__main__":
    # ------------------------- user settings -------------------------
    distribution = "parabola"   # "gaussian" or "parabola"
    convention = "z"            # "z" or "tau"  (sets the sign of h and dp/p0)
    branch = "under"            # "under" or "over" compression
    FIGS = ps.output_dir(Path(__file__).resolve().parents[1], "tools", "figures")

    # One panel per operating point. Every beam quantity is free to edit:
    #   charge          bunch charge [C]
    #   spread          correlated dp/p0 the injector delivers, SIGNED
    #                   (its sign is what fixes the sign of R56)
    #   sigma_z_before  rms bunch length before compression [m]
    #   peak_current    target peak current after compression [A]
    # Seeded with the values measured from data/OP*_50k.dist. The background is
    # drawn with the panel's own charge and target, so the iso-line at `spread`
    # passes through the solved marker by construction.
    panels = [
        dict(name="OP1", branch_label="SASE 1 THz, dogleg",
             charge=1.000e-9, spread=+0.01, sigma_z_before=1.31e-3,
             peak_current=200.0,
             sigma_z0_axis=np.linspace(0.7e-3, 1.5e-3, 200),
             r56_axis=np.linspace(-0.09, -0.04, 200),
             c_levels=[2.0, 2.5, 3.0, 3.5]),
        dict(name="OP2", branch_label="SASE 10 THz, dogleg",
             charge=1.000e-9, spread=+0.0074, sigma_z_before=1.09e-3,
             peak_current=200.0,
             sigma_z0_axis=np.linspace(0.7e-3, 1.5e-3, 200),
             r56_axis=np.linspace(-0.11, -0.05, 200),
             c_levels=[1.5, 2.0, 2.5, 3.0]),
        dict(name="OP3", branch_label="SR 1 THz, chicane",
             charge=0.200e-9, spread=-0.006, sigma_z_before=0.76e-3,
             peak_current=400.0,
             sigma_z0_axis=np.linspace(0.5e-3, 1.1e-3, 200),
             r56_axis=np.linspace(0.11, 0.20, 200),
             c_levels=[10, 15, 20]),
        dict(name="OP4", branch_label="SR 0.3 THz, chicane",
             charge=0.600e-9, spread=-0.0046, sigma_z_before=1.05e-3,
             peak_current=400.0,
             sigma_z0_axis=np.linspace(0.7e-3, 1.4e-3, 200),
             r56_axis=np.linspace(0.14, 0.24, 200),
             c_levels=[5, 7, 9]),
    ]
    ekin_ev = 15.4e6            # for the sigma_E/E0 conversion print; None = skip
    # -----------------------------------------------------------------

    # ---------------------------- compute ----------------------------
    profile = PROFILES[distribution]
    for panel in panels:
        sz_before, charge = panel["sigma_z_before"], panel["charge"]
        sz_target = profile.sigma_z_at(charge, panel["peak_current"])
        panel.update(
            sigma_z_target=sz_target,
            compression=sz_before / sz_target,
            current_before=peak_current(charge, sz_before, profile),
            r56=required_r56(sz_before, panel["spread"], sz_target,
                             convention, branch),
            scan=scan_compression(panel["sigma_z0_axis"], panel["r56_axis"],
                                  charge, panel["peak_current"],
                                  distribution=distribution,
                                  convention=convention, branch=branch))

    print(f"{distribution}, {branch}-compression, {convention} convention")
    if ekin_ev is not None:
        b2 = beta_squared(ekin_ev)
        print(f"sigma_E/E0 = {b2:.6f} x dp/p0 at E_kin = {ekin_ev/1e6:.1f} MeV "
              f"({(1-b2)*100:.2f} % lower)\n")
    print(f"{'OP':>4s} {'Q/nC':>6s} {'sig_z bef/mm':>13s} {'I_pk bef/A':>11s} "
          f"{'I_pk tgt/A':>11s} {'sig_z tgt/mm':>13s} {'C':>7s} "
          f"{'dp/p0 %':>9s} {'R56 req/mm':>11s}")
    for panel in panels:
        print(f"{panel['name']:>4s} {panel['charge']*1e9:6.2f} "
              f"{panel['sigma_z_before']*1e3:13.4f} "
              f"{panel['current_before']:11.1f} {panel['peak_current']:11.0f} "
              f"{panel['sigma_z_target']*1e3:13.4f} {panel['compression']:7.2f} "
              f"{panel['spread']*100:+9.4f} {panel['r56']*1e3:+11.1f}")

    # ----------------------------- plot ------------------------------
    ps.apply_style()
    fig, axes = plt.subplots(figsize=(8, 5.5), nrows=2, ncols=2,
                             layout="constrained")

    for ax, panel in zip(axes.flat, panels):
        scan = panel["scan"]
        spread_pct = scan.spread * 100.0
        level = panel["spread"] * 100.0

        cf = ax.contourf(scan.sigma_z0 * 1e3, scan.r56, spread_pct,
                         levels=20, cmap="plasma")
        fig.colorbar(cf, ax=ax).set_label(
            r"Required correlated $\sigma_{p_z}/p_{z0}$ [%]")

        cs_c = ax.contour(scan.sigma_z0 * 1e3, scan.r56, scan.compression,
                          levels=panel["c_levels"], colors="black",
                          linewidths=1.1, linestyles="--")
        ax.clabel(cs_c, fmt={c: f"C = {c:g}" for c in panel["c_levels"]},
                  fontsize=9)

        # iso-line at the spread this injector beam delivers
        if spread_pct.min() <= level <= spread_pct.max():
            cs_op = ax.contour(scan.sigma_z0 * 1e3, scan.r56, spread_pct,
                               levels=[level], colors="white",
                               linewidths=1.8, linestyles="-")
            ax.clabel(cs_op, fmt={level: f"{level:+.2f} %"}, fontsize=9)
        else:
            print(f"  [warn] {panel['name']}: spread {level:+.3f} % outside "
                  f"the panel range [{spread_pct.min():+.3f}, "
                  f"{spread_pct.max():+.3f}] % -- widen r56_axis")

        ax.plot(panel["sigma_z_before"] * 1e3, panel["r56"], "o",
                color="white", markersize=10, markeredgecolor="black",
                markeredgewidth=1.4, zorder=5,
                label=(rf"before: $\sigma_{{z,b}}$ = "
                       rf"{panel['sigma_z_before']*1e3:.2f} mm, "
                       rf"$C$ = {panel['compression']:.2f}" "\n"
                       rf"$R_{{56}}$ = {panel['r56']*1e3:+.1f} mm"))

        ax.set_title(f"{panel['name']} — {panel['branch_label']}\n"
                     rf"$Q$ = {panel['charge']*1e9:.1f} nC, "
                     rf"target $I_{{\rm peak}}$ = "
                     rf"{panel['peak_current']:.0f} A "
                     rf"($\sigma_{{z,b}}$ after = "
                     rf"{panel['sigma_z_target']*1e3:.2f} mm)", fontsize=11)
        ax.legend(loc="upper left", framealpha=0.85, fontsize=9)

    for ax in axes.flat:
        ax.set_xlabel(r"$\sigma_{z,b}$ before compression [$mm$]")
        ax.set_ylabel(r"$R_{56}$ [$m$]")

    plt.show()
    ps.save(fig, FIGS, 'fig_estimate_req_r56')
