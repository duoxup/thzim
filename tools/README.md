# Standalone Tools

Each script is run as `python <script>.py` with its settings edited at the top
of its `__main__` block; none takes command-line arguments. The sizing maps
save their figures under `outputs/tools/figures/` through `plt_style.py`, the
tools-side copy of `thzim.utils` (matplotlib only); the demos import
`thzim.utils` and need the package installed. These are parameter checks and
method demonstrations, not simulations.

- `plt_style.py` — `apply_style` / `output_dir` / `save`, shared by the
  sizing maps below so they run without `thzim`.

- `bunch_compression_scan.py` — sizing-stage compressor scan from linear theory:
  required correlated momentum spread over a (σ_z,b, R56) grid, with
  compression-factor contours. One panel per operating point (2×2), each
  overlaid with the measured beam — iso-line at the spread the injector
  delivers, marker at the R56 it implies. The beam values are seeded in the
  settings block; `measure_beam()` re-reads them from `data/OP*_50k.dist` and
  is the only place `partdist` is needed (imported lazily). Both sign
  conventions (z / tau) and both branches (under / over).
- `bunch_compression_r56.py` — the transpose of `bunch_compression_scan.py`:
  the correlated spread moves to the y axis and the colour map becomes the
  required R56, i.e. "the injector gives me this chirp, how long must the
  compressor be?" — the form the chicane/dogleg footprint is sized from. Same
  linear model, same 2×2 layout and free-parameter settings block; the C
  contours are vertical (C depends on the pre-compression length alone) and are
  labelled at a fixed height so they stay clear of the legend. Imports its
  physics from `bunch_compression_scan.py`, so keep the two side by side.
- `bunch_compression_yield.py` — the same model run forwards, with no target:
  the correlated spread is fixed per operating point and the map reports what a
  given R56 actually compresses the bunch to (peak current or σ_z,b after).
  Marks one user-specified working point per panel and draws the
  full-compression ridge, where the linear model diverges — the colour scale is
  capped there on purpose. Also imports its physics from
  `bunch_compression_scan.py`.
- `chicane_geometry_map.py` — sizing the four-dipole chicane: R56 over a
  (θ, L_Dz) grid at fixed dipole z-projection L_Bz. Solves and marks the θ each
  operating point's R56 target needs. Small-angle R56, no velocity term —
  1.6–2.5 % low at the design angles, deliberate.
- `dogleg_geometry_map.py` — sizing the two-dipole achromatic dogleg: |R56| over
  a (θ, ρ) grid. The achromat theorem makes R56 = −2ρ(θ − sin θ) a function of
  those two alone, so ρ inverts in closed form. Velocity term dropped (6 % at
  15.4 MeV, accepted — the dogleg's R56 only needs to be approximate).
- `chicane_offset_map.py`, `dogleg_offset_map.py` — the transverse excursion Δx
  on the same axes as the two R56 maps above, kept separate because it is a
  secondary constraint: it never changes the R56 choice, it only says whether
  that choice fits the chamber and the layout. Each imports its geometry from
  the matching `*_geometry_map.py`, so keep the pairs together. The dogleg one
  takes a reference inter-dipole straight `l_gap_ref` (set it to 0 to read the
  floor the dipoles impose on their own).
- `peak_current_map.py` — the charge/length/current relation on its own, with
  no compressor in it: I_peak over an (σ_z,b, Q) grid for the selected current
  profile, log-log by default so the iso-current lines run at 45°. Charge is
  conserved, so a compressor is the horizontal arrow drawn from each operating
  point's pre-compression length to its target. Shares the profile models with
  `bunch_compression_scan.py`.
- `undulator_resonance.py` — analytic resonance of a helical (APPLE-II)
  undulator, optionally in a waveguide: gap → B0 (exponential fit) → K_rms →
  resonant wavelength, plotted per undulator period. A parameter check for
  the working-point energies, not an FEL simulation (that is delivered
  separately). Figures are shown, not saved; only needs numpy/scipy/matplotlib.
- `demo_triplet_round_transport.py` — demo / smoke test for `thzim.triplet`.
  Scans the first-quad gradient g1 over 5 values, solves (g2, g3) for a round
  exit beam at each, and draws sigma(s) in two side-by-side panels: space
  charge off (analytic 2x2 over `thzim.maps`) and on (the SAME knobs,
  ocelot-tracked).
  Colour = g1, solid = sigma_x, dashed = sigma_y. The printed `lin dev` /
  `sc dev` columns are the worst fractional |sigma_x − sigma_y| anywhere in the
  exit drift, and the point of the figure is that the second stays small at the
  first's knobs — which is why the original study never solved g2, g3 under
  space charge. Set `SC_KNOBS = True` to test that claim (minutes per g1). Needs
  `thzim` installed plus `partdist`; builds its own Gaussian beam, so no data
  file is required.
- `demo_two_triplet_scan.py` + `demo_two_triplet_e2e.py` — demo / smoke test
  for `thzim.two_triplet`, split the same way the dogleg pair is: scan first,
  a person chooses, then simulate the choice. Case: OP1's P0 beam onto a
  dogleg-entrance-like Twiss on the development geometry (not the design of
  record; `repro/OP1/` has that).
  1. `demo_two_triplet_scan.py` scans both legs' free gradient and draws the
     leg envelopes (sigma(s) per g1, colour = g1, solid = sigma_x, dashed =
     sigma_y) and the (sigma@screen, sigma@screen) map, then TABULATES the
     candidate crossings — it does not pick one. Two screens do NOT pin a beam,
     so the curves cross more than once; the table's `free scr` column is how
     far the two legs disagree at the screen the crossing did not use, which
     separates a real match from a parabola coincidence (1.2 % against 67 %,
     delivered Bmag 1.02 against 302).
  2. `demo_two_triplet_e2e.py` takes the chosen gradients `T1_G` / `T2_G`
     (all six -- a g1 alone does not pin the branch), evaluates them with
     `match_with` and tracks the line end to end: normalised
     emittance, beta with the target marked, rms size, and the ocelot element
     plot, x blue and y red.
  Both carry an SC switch (`SC_SCAN` / `SC`); off runs analytically in seconds,
  which is how to check the g1 windows before paying for the tracked version.
  Need `thzim` installed plus `partdist`, and `data/OP1_50k.dist`.
- `demo_quadruplet_match.py` — demo / smoke test for `thzim.quadruplet`, the
  other matching route. Four quads against four exit Twiss numbers is a square
  problem, so there is no scan and nothing for a person to choose: ONE script
  that runs the linear multi-start solve, then (with `SC = True`) the
  damped-Newton re-match under space charge seeded from it, then tracks the
  result end to end and draws the same four panels as the two-triplet e2e —
  emittance, beta (tracked solid against the linear optics at the same k1
  dashed, so the SC shift is visible), rms size, ocelot element plot; x blue,
  y red. Prints the linear and SC-corrected k1 side by side with the pre- and
  post-correction Bmag. Case: the OP2 final focus, dogleg-exit Twiss onto the
  undulator round waist beta* = 0.53 m, on the OP2 P0 beam re-conditioned to
  the launch Twiss by `partdist.match_twiss_xy` (a real beam, not a synthetic
  Gaussian — the original study's demos built their own). `SC = False` is the
  seconds-long look at whether the geometry admits a solution at all. Needs
  `thzim` installed plus `partdist`, and `data/OP2_50k.dist`.
