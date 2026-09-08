#!/usr/bin/env python3
r"""Single-plane 2x2 linear transfer maps, and the covariance transport with them.

The smallest shared piece of linear optics in the package. Ocelot computes all
of this too, and where a full lattice already exists that is what the other
modules use -- `thzim.chicane` and `thzim.dogleg` both go through
`lattice_transfer_map`. This module exists for the opposite case: a solver that
evaluates one plane of a short section thousands of times inside a root find,
where building an Ocelot lattice per evaluation would dominate the cost. It is
therefore deliberately **numpy-only**; nothing here imports ocelot.

Everything is 2x2 and single-plane, in the (u, u') basis:

    drift_map(L)              [[1, L], [0, 1]]
    quad_map(k, L)            thick quadrupole, FOCUSING for k > 0
    propagate(R, S)           R S R^T          -- covariance
    propagate_twiss(R, b, a)  the same, in Twiss coordinates

**Sign convention.** `quad_map(k, L)` focuses for k > 0. One physical magnet of
geometric strength k1 focuses in x and defocuses in y, so its two planes are
`quad_map(+k1, L)` and `quad_map(-k1, L)`. Callers that carry a plane flag
usually spell this `sign = +1` for x and `-1` for y and pass `sign * k1`.

Maps compose right-to-left, as matrices do: a beam that meets A then B is
transported by `B @ A`.
"""

import numpy as np

__all__ = ["drift_map", "quad_map", "propagate", "propagate_twiss"]

K_TOL = 1e-12          # |k| below this is treated as a drift (avoids 0/0 in sqrt)


def drift_map(length):
    """2x2 map of a drift of `length` [m]."""
    return np.array([[1.0, length], [0.0, 1.0]])


def quad_map(k, length):
    """2x2 map of a THICK quadrupole, geometric strength `k` [1/m^2].

    k > 0 focuses (trigonometric), k < 0 defocuses (hyperbolic), k ~ 0 is a
    drift. Pass `-k` to get the other plane of the same magnet.
    """
    if abs(k) < K_TOL:
        return drift_map(length)
    sk = np.sqrt(abs(k))
    phi = sk * length
    if k > 0:
        return np.array([[np.cos(phi), np.sin(phi) / sk],
                         [-sk * np.sin(phi), np.cos(phi)]])
    return np.array([[np.cosh(phi), np.sinh(phi) / sk],
                     [sk * np.sinh(phi), np.cosh(phi)]])


def propagate(R, sigma):
    """Transport a 2x2 covariance block: `R sigma R^T`.

    `sigma` is [[<uu>, <uu'>], [<uu'>, <u'u'>]] in SI units [m^2, m rad, rad^2].
    The determinant (the geometric emittance squared) is invariant, since every
    map here is symplectic.
    """
    return R @ sigma @ R.T


def propagate_twiss(R, beta, alpha):
    """(beta, alpha) after the 2x2 map `R` -- `propagate` in Twiss coordinates.

    Identical transport to `propagate`; the two differ only in how the second
    moments are parameterised, `Sigma = eps [[beta, -alpha], [-alpha, gamma]]`
    with `gamma = (1 + alpha^2)/beta`. The emittance factors out, which is why
    this form needs no beam -- use it for a design Twiss, and `propagate` when
    the covariance came from real particles.
    """
    gamma = (1.0 + alpha**2) / beta
    B = R @ np.array([[beta, -alpha], [-alpha, gamma]]) @ R.T
    return B[0, 0], -B[0, 1]
