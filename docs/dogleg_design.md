# Dogleg Design

The bunch compressor of the SASE branch (OP1, OP2). A dogleg is used rather than
a chicane because that branch runs a **positive** energy chirp, which needs
`R56 < 0`; the superradiant branch (OP3, OP4) runs a negative chirp and uses a
chicane instead. The dogleg also translates the beamline axis sideways by
`Delta_x`, which is why it exists in this position at all.

Layout, mirror-symmetric about the centre marker `C`:

```
START  B1(+t)  d1  QO  d2  QI  d3 | C | d3  QI  d2  QO  d1  B2(-t)  END
```

Two quadrupole families, each sharing one strength: the outer pair `QO` (`ko`)
and the inner pair `QI` (`ki`).

## The design rests on three orthogonal knobs

This is what makes the procedure a sequence of independent solves rather than a
global optimisation:

| Knob | Set by | Controls | Leaves untouched |
|---|---|---|---|
| `rho` (with `theta`) | the R56 target | `R56` | — |
| `ko` | the achromat condition | dispersion closure | `R56` |
| `ki` | free — used to optimise | peak beta, beam roundness | `R56`, the achromat |

The orthogonality is not an approximation. Sweeping `ki` from −70 to −6 m⁻²
forces `ko` to retune over 43.9 → 67.9 m⁻², while `R56` moves by
1.3 × 10⁻⁴ mm — five orders of magnitude below the value itself. See
`repro/OP1/dogleg_ki_scan.py`, panel (b).

## Stage A — sizing: R56 target → geometry

For the mirror-symmetric achromat the dipoles alone fix `R56`; the quadrupoles
barely enter and the inter-dipole straight does not enter at all:

$$R_{56,z} = -2\rho\,(\theta - \sin\theta)$$

Dropping the velocity term makes this invertible in closed form, with no root
find:

$$\rho = \frac{|R_{56,z}|}{2\,(\theta - \sin\theta)}$$

`theta` is a free layout choice; `rho` follows. The transverse offset is a
**separate, decoupled** condition, closed afterwards by the straight between the
dipoles:

$$\Delta x = 2\rho\,(1 - \cos\theta) + L_{\rm gap}\sin\theta
\quad\Longrightarrow\quad
L_{\rm gap} = \frac{\Delta x - 2\rho\,(1-\cos\theta)}{\sin\theta}$$

`Delta_x` therefore does not feed back into `R56`, and the only feasibility
requirement is that the dipole-only floor `2 rho (1 - cos theta)` stays below the
wanted offset. In practice the straight dominates: at the design point the floor
is 0.254 m of the 2.0 m offset, so `Delta_x` is governed by `theta` and
`L_gap`, not by `rho`.

Maps: `tools/dogleg_geometry_map.py` (R56 over `theta` × `rho`) and
`tools/dogleg_offset_map.py` (`Delta_x` on the same axes).

### Why the velocity term is dropped

The sizing formula omits `-L_tot/(beta gamma)^2`. This is deliberate, not an
oversight: the dogleg's peak current is not a critical specification and is
trimmed afterwards with the injector chirp, so an approximate `R56` is enough to
place the hardware. Two further reasons make the geometric value the better
design quantity:

- **The velocity term depends on where the markers are drawn.** `L_tot` is the
  full lattice path length, lead-in and lead-out drifts included. Moving `d_in`
  from 0.2 to 0.4 m changes `R56` by 0.2 mm without touching any magnet.
- **The velocity term is energy-dependent, the hardware is not.** OP1 and OP2
  share one dogleg but sit at 15.4 and 39.4 MeV, so the same magnets deliver
  −56.1 and −59.3 mm. No single `rho` hits a common target; the geometric value
  is the one thing both operating points share.

`thzim.dogleg.rho_for_r56_exact()` includes the term when it is wanted.

## Stage B — the achromat: solve `ko`

The exit conditions are `Dx = Dx' = 0`, two constraints. Mirror symmetry plus
the antisymmetric dipole forcing collapse them to the single condition

$$D_x(C) = 0$$

so one knob suffices, and `ko` is pinned. `ki` is left completely free — the
achromat says nothing about it.

`thzim.dogleg.solve_achromat_quads(geom, ki)` matches `Dx(C) = 0` with a
**shared** `QO` instance appearing on both sides of the centre, so the match has
exactly one variable, then rebuilds the lattice with independent instances
(tracking-safe) and reports the residual `Dx`, `Dx'` at the exit.

Two properties worth keeping in mind when reading results:

- The exit dispersion is a **by-product of this stage**, available before any
  Twiss is chosen. Dispersion is driven by the dipoles from `Dx = Dx' = 0` at
  the entrance, so it never depends on the entrance beta/alpha.
- The residual is a *match tolerance*, not physics: |Dx_exit| ≈ 5 × 10⁻⁴ mm.

## Stage C — the entrance Twiss

With the lattice fixed, the question inverts: what must the upstream match
deliver? The lattice has no x–y coupling, so the entrance→centre transfer matrix
is block diagonal and the planes are solved **independently**. Per plane:

1. Place a waist at the centre (`alpha(C) = 0`) and back-propagate it,
   $\Sigma_0 = M_2^{-1}\,\mathrm{diag}(\beta_c,\,1/\beta_c)\,M_2^{-\rm T}$, giving
   $\beta_0 = \Sigma_{0,11}$ and $\alpha_0 = -\Sigma_{0,12}$.
2. Choose `beta_c` to minimise the **true** peak beta through the dogleg —
   measured on a finely sliced lattice, so a peak between elements is not
   missed.

`thzim.dogleg.solve_entrance_twiss(dl)` returns the four entrance values plus
the peak betas. Note the peak betas fall out of step 2 itself; propagating the
solved Twiss forward afterwards reproduces them exactly and derives nothing new.
`repro/OP1/dogleg_forward.py` does that propagation as a
visualisation and a self-check.

## Stage D — choosing `ki`

Since the achromat leaves `ki` free, it is spent on beam quality. Scan it, and
for each value run Stage B and Stage C; score with

$$\text{score} = \max\!\left(\beta_x^{\rm peak},\ \frac{\epsilon_y}{\epsilon_x}\,\beta_y^{\rm peak}\right)$$

and keep the smallest. Because $\sigma^2 = \epsilon\beta$, dividing through by
`eps_x` makes the two planes comparable, so the score is the worst-plane peak
beam size in units of `eps_x`.

`emit_ratio = 1.0` (the default) reduces this to `max(peak_beta_x,
peak_beta_y)`, the purely optical criterion, and keeps the whole dogleg design
independent of any particle distribution. Passing the measured ratio instead
makes this layer agree with the downstream crossing match, whose roundness
residual already weights the planes by it — at the cost of that independence.
The choice is deliberate and is left to the user.

`emit_ratio` cannot move the per-plane results of Stage C, only which `ki` wins:
scaling a plane by a constant does not shift its own argmin.

## R56 bookkeeping: geometric vs tracked

The two values quoted throughout differ by a known, exact amount:

$$R_{56}^{\rm tracked} = \frac{R_{56}^{\rm geo}}{\beta^2} + \frac{L_{\rm tot}}{\beta^2\gamma^2}$$

Verified against the Ocelot map from 15.4 MeV to 1 GeV, residual
2.9 × 10⁻⁵ mm at every energy — that residual is the achromat match tolerance,
not a missing term. `DoglegLattice.summary()` prints both values side by side to
keep them from being confused.

At the design point, `L_tot` is 3.774 m for both operating points. Their total
interface-drift allowance is the same, but OP1 places 0.2/0.1 m before/after
the magnets while OP2 places 0.0/0.3 m:

| Term | OP1 (15.4 MeV) | OP2 (39.4 MeV) |
|---|---|---|
| geometric −2ρ(θ − sin θ) | −59.993 mm | −59.993 mm |
| ÷ β² | −0.062 mm | −0.010 mm |
| + L_tot/(β²γ²) | +3.896 mm | +0.619 mm |
| **tracked (Ocelot)** | **−56.159 mm** | **−59.384 mm** |
| relative to geometric | −6.4 % | −1.0 % |

## Design of record

`theta = 40°`, `rho = 0.542 m`, `Delta_x = 2.0 m`, giving a geometric
`R56 = −60.0 mm`.
Derived geometry: `L_bend` = 0.378 m, `L_gap` = 2.717 m, `d3` = 0.958 m.

| | OP1 | OP2 |
|---|---|---|
| E_kin | 15.4 MeV | 39.4 MeV |
| `d_in` | 0.2 m | 0.0 m |
| `d_out` | 0.1 m | 0.3 m |
| dipole field | 0.098 T | 0.246 T |
| selected `ki` | −38.0 m⁻² (hand-set) | −22.0 m⁻² (scan optimum) |
| → `ko` | +59.295 m⁻² | +52.871 m⁻² |
| QO / QI gradient | +3.145 / −2.016 T/m | +7.038 / −2.929 T/m |
| entrance (βx, αx, βy, αy) | (10.801, +26.187, 1.734, +2.093) | (2.094, +9.488, 6.244, +7.159) |
| peak (βx, βy) | (10.801, 1.931) m | (11.698, 11.293) m |
| tracked `R56` | −56.159 mm | −59.384 mm |

**The optics carries no energy at all.** Ocelot's `k1` is a geometric strength
[m⁻²] and the bend is specified by angle, so the transfer matrix is the same at
15.4 and 39.4 MeV; `d_out` sits downstream of everything Stages B and C solve
and does not enter either. Set both operating points to the same `ki` and every
Twiss number above agrees **bit for bit**, and a `ki` scan run at either energy
returns the same optimum. Energy enters in exactly two places: the `R56`
velocity term and the `k1` → T/m conversion.

The two columns differ, then, for one reason only: they were built at different
`ki` — and `ki` is the one knob the achromat leaves free.

### The two operating points made opposite choices of `ki`

OP2 was built at whatever Stage D returns — `design_dogleg()` runs exactly
this scan — so its `ki = −22 m⁻²` **is**
the min-peak choice at `emit_ratio = 1`: the two planes balance there at
`peak beta = (11.70, 11.29) m`.

OP1's `ki = −38` was set by hand instead, and it is not that optimum. It sits
near the broad minimum of `peak beta_y` alone: `(10.80, 1.93) m`, a 5.6×
asymmetry. In other words the OP1 choice optimised the **vertical** plane rather
than roundness. Whether to move it to the scan optimum depends on whether that
was driven by a vertical-aperture constraint not represented in this model —
the same hardware at OP2 shows the scan value is reachable.

## Where a particle distribution enters

Nowhere in this chapter. Stages A–D are pure linear optics; no stage reads a
`.dist` file, and `emit_ratio` (optional, default 1.0) is the single place a
beam property may appear. The distribution first enters **downstream**, in the
upstream matching section that has to deliver the entrance Twiss:

```
repro/OP{1,2}/two_triplet_scan.py   -> both legs scanned with SC, the crossings tabulated
repro/OP{1,2}/two_triplet_e2e.py    -> P0 tracked to the dogleg entrance at the chosen gradients
```

`thzim.two_triplet` re-runs Stages B–C at the same `ki`, so the target is
computed from the dogleg definition rather than copied between scripts.

That makes the dogleg design reproducible without the beams, which is why these
scripts run against the repository alone.

## Script map

| Step | Where | What it does |
|---|---|---|
| A | `tools/dogleg_geometry_map.py` | R56 over (θ, ρ); solves ρ for an R56 target |
| A | `tools/dogleg_offset_map.py` | Δx on the same axes; the secondary constraint |
| B–D | `repro/OP{1,2}/dogleg_ki_scan.py` | scans `ki`; per value solves `ko` and the entrance Twiss, scores, recommends |
| B–C | `repro/OP{1,2}/dogleg_forward.py` | re-solves at the chosen `ki` and propagates the Twiss through, with a self-check |
| next | `repro/OP{1,2}/two_triplet_scan.py`, `two_triplet_e2e.py` | the upstream section that has to DELIVER the entrance Twiss (`thzim.two_triplet`) |
| — | `src/thzim/dogleg.py` | all of the physics; the scripts above only call and plot |

## Open items

- The `rho` values 0.577 (map-based) and 0.458 (recalibrated on tracked
  currents) appeared in earlier iterations of this design and may survive in
  old notes; the design of record is 0.542 m and no file in this repository
  quotes the others.
- OP1's hand-set `ki = −38` against the scan optimum: whether a vertical
  aperture drove it is not recorded. The upstream match now exists for both
  points (`two_triplet_scan.py`), so moving OP1 to the optimum is a re-run of
  that chain, not a redesign.
