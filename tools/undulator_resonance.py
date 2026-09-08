#!/usr/bin/env python3
"""FEL resonance calculator for a helical (APPLE-II) undulator, optionally
inside a waveguide.

Computation chain, per undulator period lambda_u:

    gap -> B0        exponential fit  B0 = a exp(-b g/lambda_u + c (g/lambda_u)^2)
        -> K_rms     e B0 lambda_u / (2 pi m_e c)
        -> lambda_r  waveguide-modified helical resonance

The resonance follows from the intersection of the beam line
omega = (k_z + k_u) c beta_z with the waveguide dispersion
k_z = sqrt((omega/c)^2 - k_c^2), giving two branches

    omega_r = c k_u beta_z gamma_z^2 [1 + direction * sqrt(1 - (k_u^2 + k_c^2)
                                                           / (k_u^2 gamma_z^2))]

selected by ``direction`` (+1: v_g > v_e, -1: v_g < v_e). Setting k_c = 0
recovers the free-space helical resonance.

Layout: physics functions -> gap-scan computation (GapScan) -> ``__main__``
(user settings, compute, plot). Standalone tool: edit the user settings at the
top of the ``__main__`` block and run. Figures are shown, not saved.
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.constants import c as c_light, e as q_e, m_e


# --------------------------- physics ---------------------------

def gamma_from_kinetic_energy(ke_ev):
    """Lorentz gamma from kinetic energy in eV."""
    return 1.0 + np.asarray(ke_ev) * q_e / (m_e * c_light**2)


def calc_b0(gap, lambda_u, a, b, c):
    """Peak field B0 [T] from gap [m] via the exponential gap fit."""
    gap = np.asarray(gap)
    return a * np.exp(-b * gap / lambda_u + c * (gap / lambda_u) ** 2)


def calc_krms(b0, lambda_u):
    """RMS undulator parameter K_rms = e B0 lambda_u / (2 pi m_e c)."""
    return q_e * np.asarray(b0) * lambda_u / (2 * np.pi * m_e * c_light)


def calc_gamma_z(gamma, krms):
    """Longitudinal gamma_z = gamma / sqrt(1 + K_rms^2)."""
    return np.sqrt(np.asarray(gamma) ** 2 / (1 + np.asarray(krms) ** 2))


def beta_from_gamma(gamma):
    return np.sqrt(1 - 1 / np.asarray(gamma) ** 2)


def calc_lambda_r_vacuum(gamma, lambda_u, krms):
    """Free-space helical resonance lambda_r = lambda_u (1 + K_rms^2) / (2 gamma^2)."""
    return lambda_u / (2 * np.asarray(gamma) ** 2) * (1 + np.asarray(krms) ** 2)


def calc_lambda_r_helical(gamma, lambda_u, krms, direction=1, kc=0.0):
    """Waveguide-modified helical resonance wavelength [m].

    Branches below cutoff (negative sqrt argument) return NaN.
    """
    k_u = 2 * np.pi / lambda_u
    gamma_z = calc_gamma_z(gamma, krms)
    beta_z = beta_from_gamma(gamma_z)
    omega_r = c_light * k_u * beta_z * gamma_z**2 * (
        1 + direction * np.sqrt(1 - (k_u**2 + kc**2) / (k_u**2 * gamma_z**2))
    )
    return 2 * np.pi * c_light / omega_r


# ---------------------- gap-scan computation ----------------------

@dataclass
class GapScan:
    """All gap-dependent quantities for one undulator period."""

    lambda_u: float     # undulator period [m]
    gaps: np.ndarray    # gap values [m]
    b0: np.ndarray      # peak field [T]
    krms: np.ndarray    # RMS undulator parameter
    lam_lo: np.ndarray  # resonance wavelength [m] at min kinetic energy
    lam_hi: np.ndarray  # resonance wavelength [m] at max kinetic energy

    @property
    def label(self):
        return f"$\\lambda_u$={self.lambda_u*1e3:.0f} mm"


def scan_gap(lambda_u, gaps, a, b, c, ekin_range_ev, direction=1, kc=0.0):
    """Evaluate B0, K_rms and the resonance band over a gap range."""
    b0 = calc_b0(gaps, lambda_u, a, b, c)
    krms = calc_krms(b0, lambda_u)
    gammas = gamma_from_kinetic_energy(np.asarray(ekin_range_ev))
    lam_lo = calc_lambda_r_helical(gammas.min(), lambda_u, krms, direction, kc)
    lam_hi = calc_lambda_r_helical(gammas.max(), lambda_u, krms, direction, kc)
    return GapScan(lambda_u, np.asarray(gaps), b0, krms, lam_lo, lam_hi)


if __name__ == "__main__":
    # ------------------------- user settings -------------------------
    # B0(gap) fit coefficients: B0 = a exp(-b g/lambda_u + c (g/lambda_u)^2)
    a, b, c = 1.54, 4.46, 0.43
    # undulator periods to compare [m], each with its own gap range [m]
    undulators = [
        (65e-3, np.linspace(18e-3, 40e-3, 2001)),
        (130e-3, np.linspace(30e-3, 60e-3, 2001)),
    ]
    ekin_range_ev = (15e6, 40e6)  # kinetic energy band edges [eV]
    kc = 0.0        # waveguide cutoff [rad/m]; e.g. TE11 circular pipe:
                    #   kc = 1.8412 / r_wg; 0 = free space
    direction = +1  # resonance branch: +1 (v_g > v_e) or -1 (v_g < v_e)
    # annotation positions on the wavelength panel: (gap [mm], lambda [um])
    band_labels = [(30, 30), (45, 500)]
    wl_yticks = [10, 30, 100, 300, 1000, 3000]  # wavelength axis ticks [um]
    # -----------------------------------------------------------------

    # ---------------------------- compute ----------------------------
    scans = [scan_gap(lambda_u, gaps, a, b, c, ekin_range_ev,
                      direction=direction, kc=kc)
             for lambda_u, gaps in undulators]

    # ----------------------------- plot ------------------------------
    band_color = "#1ECBE1" if direction == 1 else "#E1341E"

    with plt.rc_context({"font.size": 14}):
        fig, (ax_b0, ax_k, ax_wl) = plt.subplots(
            figsize=(12, 4), ncols=3, layout="constrained")

        for scan in scans:
            gaps_mm = scan.gaps * 1e3
            ax_b0.plot(gaps_mm, scan.b0, label=scan.label)
            ax_k.plot(gaps_mm, scan.krms, label=scan.label)
            ax_wl.plot(gaps_mm, scan.lam_lo * 1e6, color="#E1341E")
            ax_wl.plot(gaps_mm, scan.lam_hi * 1e6, color="#1ECBE1")
            # fill_between cannot handle NaN (below-cutoff points): pad with
            # the first valid value so the shading spans the full gap range
            first_valid = scan.lam_lo[np.where(~np.isnan(scan.lam_lo))[0][0]]
            lam_lo_filled = np.nan_to_num(scan.lam_lo, nan=first_valid)
            ax_wl.fill_between(gaps_mm, scan.lam_hi * 1e6, lam_lo_filled * 1e6,
                               alpha=0.3, color=band_color)

        ax_b0.set_ylabel("$B_0$ [$T$]")
        ax_b0.legend()

        ax_k.set_ylabel("$K$")
        ax_k.legend()

        ax_wl.set_yscale("log")
        ax_wl.set_ylabel("$\\lambda_r$ [$\\mu m$]")
        ax_wl.set_yticks(wl_yticks)
        ax_wl.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
        # the two labels map to the first scan's lower/upper energy curves;
        # the colors are shared by all scans
        ax_wl.legend([f"$E_0$={np.min(ekin_range_ev)/1e6:g} MeV",
                      f"$E_0$={np.max(ekin_range_ev)/1e6:g} MeV"],
                     loc="upper right")
        for scan, (x_text, y_text) in zip(scans, band_labels):
            ax_wl.text(x_text, y_text, scan.label, ha="center", va="center")

        for ax in (ax_b0, ax_k, ax_wl):
            ax.set_xlabel("$g$ [$mm$]")
            ax.grid(True)

    plt.show()
