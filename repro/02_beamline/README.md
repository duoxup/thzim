# Stage 2 — Middle Optics (P0 → P3)

Tracks each working point from booster2 exit (P0) through matching sections
and compressor (chicane or dogleg, depending on branch) to the undulator
entrance (P3) with Ocelot SC + CSR physics.

Interface planes: P0 booster2 exit, P1 compressor entrance, P2 compressor
exit, P3 undulator entrance.

## Contents

- `OP{1..4}/` — per-operating-point design scripts, one set per compressor.
  The lattice geometry and selected magnet strengths live in each executable
  script's settings block; there is no separate knob table to keep in sync.

  **SASE branch, dogleg (OP1, OP2)** — pure linear optics over `thzim.dogleg`,
  reading no distribution; the geometry comes from
  `tools/dogleg_geometry_map.py`. Run in this order:
  1. `dogleg_ki_scan.py` — the achromat leaves the inner quad pair ki free.
     Scan it: each ki gives a ko (achromat), an entrance Twiss and the peak
     betas that go into the score. Pick a ki from the result.
  2. `dogleg_forward.py` — re-solve at that ki and propagate the Twiss through,
     to see and verify the lattice it produced.

  OP1 and OP2 share the dogleg hardware and, because Ocelot's `k1` is geometric
  and the bend is given by angle, its optics as well — the two scans agree
  column for column. They differ in entrance/exit interface-drift allocation,
  in the gradients in T/m, in the R56 velocity term, and in `ki`: OP1's −38
  was hand-set, OP2's −22 is the scan optimum. See `docs/dogleg_design.md`.

  Then the section that DELIVERS that entrance Twiss, over
  `thzim.two_triplet` — two quad stations with the switched-off chicane
  footprint between them. Its three screens are fixed at 10 %, 50 % and 90 %
  of the 3.40 m inter-station gap, and the match is selected where the two
  screen-size curves cross. Split the same way, because the choice is a
  person's:
  3. `OP{1,2}/two_triplet_scan.py` — scans both legs' free gradient and
     tabulates the candidate crossings. It does not pick one. Two screen
     readings do not pin a beam, so some crossings can be spurious; the
     independent middle-screen `free scr` residual separates them. The
     envelopes in figure 1 show which candidates pinch to a hard waist in the
     middle — at 1 nC that is where the emittance goes.
  4. `OP{1,2}/two_triplet_e2e.py` — takes the chosen GRADIENTS (all six: a g1
     alone does not determine the lattice, the round solve has several
     branches) and tracks P0 → dogleg with SC, against the target and against
     all three screen readings. The settings block retains both the linear
     crossing and the active SC settings. An SC run writes
     `outputs/OP<n>/beams/p1.ast` for the compressor stage.

  OP2 runs the same section at 2.5x the rigidity with a P0 beam that arrives
  converging at 0.10 mm instead of diverging at 2.3 mm, so its gradients are
  ~10x OP1's and its backward leg is steep (the scan steps it by 0.02 T/m).
  With its entrance marker at the first dogleg bend (`d_in = 0`), OP2's useful
  crossing is on the negative T2 branch near `g1_bwd = -5.6 T/m`. Its selected
  SC crossing has a 0.42 % independent-screen residual and 2.0 % roundness
  deviation; re-solving the round knobs under SC is unnecessary.

  The target comes from step 2, so the four scripts are one chain: dogleg
  design → entrance Twiss → upstream match.

  **Superradiant branch, chicane (OP3, OP4)** — a chicane is achromatic by
  symmetry, so only the entrance Twiss is left, and it splits in two: `y` is an
  edge-focusing channel solved exactly by `thzim.chicane.matched_y`, while `x`
  is exactly a drift and its `beta_x` is settled only by space charge.
  - `chicane_beta_x_scan.py` — pins the matched `y`, scans `beta_x` with SC +
    CSR over `data/OP{3,4}_50k.dist`, and draws the beta / size / emittance /
    bunching evolution plus the three exit objectives in one 3-D scatter with
    the Pareto-optimal points marked. The scan is used as a soft engineering
    trade-off; `beta_x = 30 m` is the adopted nominal value for both OP3 and
    OP4.
  - `m1_quadruplet_match.py` — solves the P0 → P1 quadruplet match with
    `thzim.quadruplet`. The target is `beta_x = 30 m`, `alpha_x = 0` (the
    nominal choice from the scan above) and the analytic `matched_y` for the
    OP's own chicane geometry with `lead = 0.2 m`. Runs linear solve + SC
    re-match, then tracks the matched line end to end. An SC run writes
    `outputs/OP<n>/beams/p1.ast`.

  OP4 is the collective-dominated point (0.6 nC at 21.8 MeV against OP3's
  0.2 nC at 39.9 MeV): the trade-off has the same shape and the same `beta_x`
  wins, but it is worth several um of emittance instead of one.
- P0 → P3 driver script — entry point running the per-OP line
  (to migrate, see MIGRATION.md)

Input: `data/OP{1..4}_50k.dist`. Output: per-segment beam dumps and the
`p3.ast` beams consumed by `repro/03_fel/`.

## Run

<!-- TODO: driver command per OP -->
