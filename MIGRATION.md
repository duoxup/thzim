# Migration Checklist

Source repository: `py4pitz`. No local checkout path is part of this project.
Check off each item as it is migrated. Rules that apply to every code file:

- replace `sys.path.insert(...)` cross-imports with `from thzim import ...`;
- remove `xtils` usage (`new_subplots`, `save_figure_auto_date` → `thzim.utils`);
- fix paths to the new layout (`data/`, `repro/`);
- no absolute paths (`/home/...`, `/afs/...`, `/lustre/...`);
- **rename `case<n>` → `OP<n>`** (operating point) in identifiers, filenames,
  CLI arguments and prose. The mapping is one-to-one: `case1`→`OP1` … `case4`→`OP4`.
  py4pitz source paths quoted below keep their original `case*` names.

## Toolchain — `src/thzim/`

- [x] `chicane.py` ← `chicane_design/no_quad/chicane_study_lib.py`. Linear layer
      done: footprint geometry, the `ChicaneGeom` dataclass,
      `chicane_seq`/`build_lattice`, `r56_z_exact` and `matched_y` (verified
      bit-identical to the original). Changes: `ChicaneGeom` is new, mirroring
      `DoglegGeom` — note the two are constrained differently, the dogleg by
      offset (rho, Delta_x given) and the chicane by z-footprint (L_Bz, L_Dz
      given) — and the lattice/solver functions now take it instead of loose
      arguments; the
      bookkeeping drifts `LEAD`/`TAIL` are explicit parameters — they are not
      cosmetic, `lead` shifts `beta*` by ~5 % and flips the sign of `alpha*`
      (documented in the docstring), and exit self-reproduction needs
      `lead == tail`. The project entrance-plane convention is `lead=0.2 m`;
      `tail` remains zero unless a downstream interface is explicitly needed.
      `design_entrance` and
      `BX_SOFT` deliberately NOT migrated: that wrapper bundled the y hard solve
      with the free x choice, which the two-layer split makes misleading —
      `beta_x` belongs to a tracking scan, not to a linear solver.
      The SC/CSR tracking layer (`track_chicane`, `Monitor`) went to
      `compressor.py` (below), generalised to any lattice so the dogleg shares
      it; the `beta_x` scan is `repro/OP{3,4}/chicane_beta_x_scan.py`.
      Beam loading is partdist's job. Nothing is left to migrate here.
- [x] `compressor.py` ← the tracking half of `chicane_design/no_quad/
      chicane_study_lib.py` (`track_chicane`, `Monitor`, `emit_proj`,
      `emit_corr`, `peak_current`) and the `track_seg` / `report` pair of
      `beamline_design/segments.py`. One `track_compressor(lattice, dist, sc,
      csr)` for BOTH compressors: the chicane and the dogleg ask the same
      questions of a fixed lattice, so a per-compressor tracker would have
      been the same code twice. `BeamMonitor` records at every navigator step
      (sizes total and betatron, statistical eta, dispersion-corrected eps_x,
      eps_y, sigma_z, sigma_dp, mean dp, peak current) and adds the exit plane
      explicitly, since the navigator's last step stops short of the end
      marker. `beam_report` is the scalar exit summary (the same quantities
      plus Twiss and chirp). The partdist -> ocelot boundary is
      `triplet.as_ocelot`, so the tau-centring fix applies here too. Verified
      on OP3 against the `chicane_beta_x_scan.py` convergence table at
      beta_x = 30 m (2.63 um corrected eps_x and 14 mm residual eta here
      against 2.61 um / 12.4 mm there on the conditioned beam).
- [x] `record.py` — new; the design of record as importable data (see the
      `run_line.py` entry under Reproduction). `DoglegKnobs` re-solves `ko`
      from `ki` rather than storing it; everything else is a literal. Each
      section also carries the SC tracking settings it was designed under
      (two-triplet sections 31^3 / 0.05 m, everything else 63^3 / 0.02 m).
      Two things had to be right before the driver reproduced the per-OP
      chain, both found on OP2, whose compression sits near its knee and
      feels the transverse conditions through space charge: (1) the SC
      settings -- tracking M1 at 63^3 / 0.02 m instead of 31^3 / 0.05 m gave
      335 A instead of 213 A at P2; (2) the energy that converts the
      two-triplet gradients to k1 -- `match_with` uses the BEAM's energy
      (40.05 MeV), and the nominal 39.91 MeV (0.35 % off) still left 241 A
      and moved P1 beta_x 1.78 -> 1.41 m. With both as the e2e script had
      them every plane agrees to the printed digits (the compressor stage
      alone reproduced `dogleg_track.py` bit for bit throughout). So the
      SASE M1 crossings are sensitive at the level that matters for the peak
      current, to both the SC mesh and a 0.3 % gradient error (a mesh
      convergence study on the 50k demo beams would not be meaningful; it
      belongs to the full-statistics runs). Remaining, definitional
      difference: the driver
      reports dispersion-corrected Twiss, the match scripts ocelot's
      projected one, so at P3 the driver's Bmag_x reads 1.003 (OP1) and
      1.010 (OP2, where beta_x* = 0.53 m and the 7 mm residual eta counts)
      against the scripts' 1.000.
- [x] `quadruplet.py` — `QuadrupletGeom.d_inter` now also takes a tuple of
      three (Q1-Q2, Q2-Q3, Q3-Q4), so the superradiant M3 -- T1.Q3 plus the T2
      triplet across the 0.92 m switch region, gaps (0.92, 0.30, 0.30) -- is
      the same solver as M1. A scalar `d_inter` is unchanged (regression: the
      OP3 M1 solve reproduces its k1 exactly). Found while doing that: the
      min-peak-beta tie-break is DEGENERATE when the peak sits at the exit
      (every exact solution then has peak beta_x = the 30 m target), so
      rounding the launch Twiss to three decimals lands on a different but
      equivalent solution; not a defect, but do not expect k1 to be stable
      under small changes of the inputs there.
- [x] `quadruplet.py` ← `match/four_quads_match_api.py`. Verified **bit-identical**
      to the original on two cases (k1, exit Twiss and peak beta all differ by
      exactly 0; gradients by 1e-12, from `scipy.constants` replacing the
      hardcoded m_e / c). Changes: `MatchGeom` → `QuadrupletGeom` (gained
      `element_spans()` and `summary()`), `MatchSolution` → `QuadrupletMatch`,
      `build_match_lattice` → `build_lattice`; `_exit_twiss_linear` and
      `_peak_beta` were made public; `_prop2` was replaced by
      `maps.propagate_twiss`; `track_match` / `sc_rematch` take partdist and
      convert internally (`sc_rematch` converts ONCE and reuses the array across
      its many tracks). `make_matched_beam` deliberately NOT migrated — it
      manufactures a beam, which belongs on the script side. Also confirmed
      here: the linear layer is energy-free, exactly as in `dogleg.py`.
- [x] `two_triplet.py` ← `match/triplet_crossing_api.py` (the working core of the
      `match/triplet_crossing_*.py` family), built on `triplet.py`. New:
      `TwoTripletGeom` keeps everything in PHYSICAL order and derives the
      reversed backward leg itself, so the upstream footgun — the backward
      leg's "g1" being T2's physical Q3, reordered by hand in `_track_e2e` — is
      handled once inside the class; `TwoTripletMatch` reports physical-order
      gradients. `backward_beam()` now conditions the INJECTION distribution with
      `partdist.match_twiss_xy` (alphas negated) instead of building a synthetic
      Gaussian, so both legs share an emittance and a longitudinal profile
      exactly. `leg_scan()` drops g1 values with no round solution to `nan`
      rather than returning a fabricated screen size — without this the crossing
      can be found on a meaningless segment (observed on a wide test window).
      The SC policy is baked in as documented results, not defaults: leg scans
      SC-on for BOTH legs, e2e SC-on.
      "Round knobs usually remain linear" survives measurement — 1.6 % worst
      roundness deviation at 39.5 MeV through 2.1 m and 4.5 % at the selected
      OP1 crossing — but it is now a
      CHECKED choice rather than an assertion. `leg_scan` reports
      `sc_roundness_dev` per point, read off the track it already ran so it
      costs nothing, and `sc_knobs=True` (threaded through `match_at` and
      `solve_crossing`) re-solves g2, g3 under space charge, seeded from the
      linear answer as `solve_round_triplet` has always done on that path.
      `TwoTripletMatch` carries `round_dev` and flags it in `summary()`.
      That diagnostic earned its place immediately: the first OP1 run reported
      13–27 % and the leg envelopes grew and then shrank inside a drift, which
      no repulsive force can do. The cause was the partdist → ocelot
      conversion, not the knobs — see the `as_ocelot` entry below.
      **Found while writing the demo, and fixed here: two screens do not pin a
      beam.** In a drift sigma^2(s) is a parabola, so at fixed emittance the
      beam still carries two free parameters and a different (waist size, waist
      position) can reproduce both readings — the curves cross more than once
      and the extra crossings are spurious, not alternative matches. On the OP1
      case the two deliver Bmag 1.02 and Bmag 302. `solve_crossing` therefore
      scans EVERY screen, forms the crossing from `screen_pair`, and picks the
      candidate the remaining screen agrees with (`pick="consistent"`, the
      default), re-evaluating both legs at the candidate's g1 rather than
      interpolating the scan — interpolation read 17 % where the true residual
      is 1 %. The upstream API took the first crossing silently; here that
      happens to be the right one, and in general it is not. The residual is
      reported and also measures scan resolution (1.2 % at 13 points per leg,
      0.36 % at 31, with Bmag following it down).
      `match_at()` is new: it builds the match at a CHOSEN (g1_fwd, g1_bwd)
      with no scan and no search, and `solve_crossing()` is now written on top
      of it. That is what lets the demo split into a scan script and an
      end-to-end script with the choice in between, the same shape as the
      dogleg pair — the upstream `run_crossing()` did scan, pick and validate in
      one call, which hid the decision.
      The fixed-seed rough edge noted earlier turned out to matter and IS fixed,
      in `leg_scan` rather than in the solver. `solve_round_triplet`'s claim
      that one seed always finds the round branch holds on the geometry it was
      written for and not on OP1's real section, where the default seed misses
      solutions that plainly exist (at g1 = -0.35 T/m it reports dev 3.6e-1
      while seed (+1, 0) reaches 1.9e-3) — and the hole it punched swallowed
      the correct crossing, leaving only a 178 %-inconsistent one. `leg_scan`
      now tries up to three seeds per point (the previous point's answer, the
      given seed, its mirror) and keeps the roundest; `solve_crossing` seeds
      each candidate's re-evaluation from the nearest scan point, and
      `match_at` takes a separate seed per leg for that. Two further fixes from
      the same run: picking with `np.argmin` over residuals containing `nan`
      returned the `nan` (an unusable candidate) — now `nan` scores as
      infinite; and a candidate whose leg has no round solution is reported as
      UNUSABLE rather than "unchecked".
- [x] `triplet._bounded_refine()` — new; the space-charge stages of
      `solve_round_triplet(sc=True)` are now BOUNDED root finds around the
      linear answer instead of unbounded `fsolve` calls. Unbounded, on an
      objective that tracks the beam, `fsolve` probes a gradient large enough
      that a particle's transverse angle exceeds its own momentum; ocelot's
      `xxstg_2_xp_mad` then square-roots a negative number, the NaN reaches the
      space-charge mesh index, and `np.bincount` raises "'list' argument must
      have no negative elements" from inside `sc.py`. It killed the first OP1
      point of a `sc_knobs=True` scan. The refinement is a few per cent
      correction to a placed answer, so the bracket starts at 5 % of
      |g_linear| and doubles to at most 100 %, a point where the track fails
      counts as outside the domain rather than as a value, and a stage that
      finds no sign change leaves the gradient at its linear value rather than
      forcing one. Verified on the point that crashed: completes, g2 held at
      its linear +4.9482 and g3 refined +0.0206 → +0.0161, `converged=False`
      because that g1 genuinely has no round solution, which `leg_scan` then
      drops to `nan` as designed. Cost there was 157 s, so the "~1 min per
      point" note has been softened.
- [x] `match_with()` — new, and the fix for a real defect in the two-script
      split. The scan hands a working point to the end-to-end script, and
      passing the two scanned gradients was NOT enough: the round solve at a
      fixed g1 has several branches, the scan picks one by continuity from its
      neighbours, and a cold `match_at` at the same g1 has nothing to be
      continuous with. On OP1 at g1_bwd = +0.3542 the scan gives T2 = (−0.579,
      +0.582, +0.354) T/m and a cold solve gives (+0.357, +5.648, +0.354) — a
      106 1/m² quadrupole, free-screen residual 180 % against 0.00 %, Bmag
      5741/423 against 1.008/1.001, and sigma_y blowing up to 35 mm past T2.
      Both are genuinely round (linear roundness deviation 1.2e-3 for either),
      so nothing but the free screen distinguishes them. `match_with` evaluates
      GIVEN gradients with no solve at all, the scan scripts print all six ready
      to copy, and the e2e scripts take `T1_G` / `T2_G` instead of `G1_FWD` /
      `G1_BWD`. `TwoTripletMatch.lin_round_dev` reports whether what was handed
      in is round, so a typo is caught.
- [x] `repro/OP{1,2}/two_triplet_scan.py`, `two_triplet_e2e.py` —
      new; the upstream match into the dogleg on the real section, split scan /
      choose / simulate. The target is computed from `thzim.dogleg`; geometry
      and selected gradients are retained directly in the scripts. The three
      screen positions are fixed at 10/50/90 % of the 3.40 m gap
      (0.34/1.70/3.06 m from its entrance). They may be fine-tuned before
      beamline construction without changing the matching method.
      OP1's linear and SC-corrected crossings are both retained in the e2e
      settings, with the SC result active by default.
      OP2 uses the same hardware at 39.4 MeV and ki = −22. Its P0
      beam arrives CONVERGING at 0.10 mm (own waist 0.094 mm at s = 0.14 m,
      before Q1), and its dogleg entrance marker is at the first bend
      (`d_in=0`). The useful crossing is on a negative T2 branch near
      `g1_bwd=-5.6 T/m`; the selected SC solution has a 0.42 % independent-
      screen residual and 2.0 % roundness deviation. Enabling `SC_KNOBS` did
      not materially change the result, so the linear round solve remains the
      default.
- [x] `matching.py` — **dropped as a name.** The two routes above are separate
      modules; a single `matching.py` would have hidden that they are different
      kinds of procedure (model vs operational), not two functions of one solver.
- [x] `crossing_match_api.py` — **dropped, not migrated.** Superseded by
      `two_triplet.py`; the name `CrossingGeom` is not reused, and neither is
      `solve_crossing`'s old signature.
      Correcting an earlier note here: it was NOT a different idea. Its
      `solve_crossing` already solved six knobs against four exit-Twiss
      residuals **plus two round-transport residuals at the ends of the middle
      drift**, emittance-weighted — the same physics `two_triplet.py`
      implements. What changes is the method: a multi-start joint least squares
      keeping "the min-peak-beta root" becomes a scan of one measurable
      quantity per leg. That matters because the problem genuinely has several
      solutions, and the crossing route puts them on a figure to choose from
      (including which one avoids a tiny waist in the middle) instead of
      picking one silently.
      This was a design change, not only a port: the earlier
      `beamline_design/segments.py` ran the retired solver. OP1 and OP2 now use
      the two-triplet crossing route.
- [x] `match/triplet_crossing_{linear,solve,match,demo,compare,mirror,scsc}.py`,
      `triplet_g1_scan{,_bwd}.py`, `triplet_match_real.py` — study scripts, not
      migrated; their conclusions are recorded in the `two_triplet.py`
      docstring. Two are worth restating: `_scsc.py` found that re-solving the
      round knobs under space charge does not pay for itself, and `_mirror.py`
      found that running the backward leg linearly and reflecting its SC
      displacement does NOT give a better crossing — both legs must be scanned
      with space charge on.
- [x] `triplet.as_ocelot()` — the partdist → ocelot boundary, and a trap worth
      recording. Two things must be right before an ocelot `SpaceCharge` track
      means anything, and getting one of them right is worse than getting
      neither, because the result still looks plausible.
      (A) `to_ocelot_particle_array(dist, s=...)` builds `tau = s - z` with `s`
      defaulting to 0, so a bunch recorded at z = 7.4 m arrives with a mean
      `tau` of −7.4 m. The partdist round trip is still exact (`pa.s` records
      the `s` used), so nothing catches it — but `SpaceCharge` converts to MAD
      constant-time coordinates by drifting each particle back by its own tau,
      which shears the bunch the field solver sees (sigma_x 2.274 → 3.803 mm)
      and, worse, re-applies the same 7.4 m lever arm to the KICKED momentum on
      the way back: a +1e-6 rad outward kick came back as −7.29 um INWARD.
      Fixed by passing the distribution's charge-weighted mean z as `s`, which
      centres `tau` and keeps `pa.s` true, so the round trip stays lossless.
      (B) partdist used to copy ASTRA's NEGATIVE macro-charge, and ocelot builds
      its density straight from `q_array`, so the bunch attracted itself. Fixed
      in partdist (`to_ocelot_particle_array` now returns `abs(Q)`;
      `from_ocelot_particle_array` restores the sign, round trip verified);
      `as_ocelot` keeps a tripwire for an older partdist.
      Verified against ocelot's own ASTRA adaptor: 3 m drift from a 2.2785 mm
      waist gives sigma_x = 2.3900 mm against its 2.3901 mm. Affected everything
      that fed a partdist distribution to an SC track — `solve_round_triplet
      (sc=True)`, all of `two_triplet`, `quadruplet`'s SC layer. NOT affected:
      `repro/OP{3,4}/chicane_beta_x_scan.py`, which go through
      ocelot's own ASTRA adaptor, and `tools/demo_triplet_round_transport.py`,
      whose synthetic beam is centred with a positive charge — which is exactly
      why the demo looked healthy and only the real `.dist` broke.
- [x] `triplet.py` ← `match/triplet_round.py`. Verified against the original
      over a g1 scan: gradients agree to 7e-9 (fsolve tolerance; the underlying
      maps are bit-identical and `brho` differs by 1e-13 relative, from
      `scipy.constants` replacing the hardcoded m_e / c), `roundness_dev` to
      3e-10, `injection_sigma` exactly. Changes: the 2x2 maps moved out to
      `maps.py` and were made public; `_core_R(…, sign)` became
      `core_R(geom, k, plane)` taking the strengths as one triple;
      `roundness_dev` gained an `l_exit` override (the crossing legs check
      roundness across the middle region, not across `drifts[3]`) and dropped
      its unused energy argument; `TripletGeom` gained `L_exit`, `s_q3_exit`,
      `length`, `element_spans()` and `summary()`, mirroring `DoglegGeom` /
      `ChicaneGeom`; `brho` / `k_of_g` / `g_of_k` come from `utils.py`.
      Deliberately NOT migrated: `gaussian_beam()` (a synthetic-beam helper —
      belongs on the script side), `envelope()` (scripts can build it from
      `maps.py` or read Ocelot's twiss output directly) and `_as_ocelot()`
      (the SC path converts internally). The entry points were also NARROWED
      to partdist only, dropping the ocelot duck-typing — see the package
      convention in `src/thzim/__init__.py`; scripts convert with
      `partdist.from_ocelot_particle_array`. Verified the narrowing changes
      no numbers, and that an ocelot round trip reproduces the solve to 4e-12.
- [x] `maps.py` — new. Single-plane 2x2 transfer maps (`drift_map`,
      `quad_map`, `propagate`), lifted from the private `_drift2` / `_quad2`
      in `match/triplet_round.py` and verified bit-identical to them. Exists
      so a root find can evaluate one plane of a short section thousands of
      times without building an Ocelot lattice; numpy-only by design.
- [x] `dogleg.py` ← `dogleg_design/dogleg_api.py`. Verified bit-identical to the
      original (ko, entrance Twiss, R56, gradients; max diff 1e-12, from
      `scipy.constants` replacing the hardcoded m_e / c). Changes:
      `design_dogleg()` gained `emit_ratio` (default 1.0 = original behaviour,
      see `EntranceTwiss.peak_score`); the vectorised sizing helpers used by
      `tools/dogleg_*_map.py` were added at module level; `rho_for_r56()` is now
      the geometry-only closed form and the velocity-term version is
      `rho_for_r56_exact()` (both take theta in RADIANS, where the original took
      degrees and put the target last).
- [x] `segments.py` — **dropped, not migrated.** `beamline_design/segments.py`
      was a per-case driver script, not a library: it hardcoded the four
      working points, called the design APIs in order and dumped beams. The
      selected design values and the ordering it encoded now live in the
      per-OP reproduction scripts and design notes. Nothing was left for a
      module.
- [x] `utils.py` — holds `apply_style` / `output_dir` / `save` (the package-side
      copy of `tools/plt_style.py`, duplicated on purpose so the standalone
      sizing tools stay runnable without installing thzim; the demo tools
      import `thzim.utils`) and the beam
      rigidity helpers `M_E_GEV` / `brho` / `k_of_g` / `g_of_k`, consolidated
      here from the separate copies in `chicane.py`, `dogleg.py` and
      `match/triplet_round.py` (the first two re-export them, so their public
      API is unchanged). matplotlib is imported lazily so the rigidity helpers
      stay cheap. The xtils helpers (`new_subplots`, `save_figure_auto_date`)
      turned out not to be needed: nothing in the package uses them.
- [x] `__init__.py` — exposes the submodules and the few cross-cutting,
      collision-free names every script imports (`apply_style` / `output_dir`
      / `save`, `as_ocelot` / `beam_energy_gev`, `track_compressor` /
      `beam_report`, `RECORDS` / `get_record`, `M_E_GEV` / `brho`). The
      geometry classes, `build_lattice`, `element_spans` and `bmag` collide
      across modules and stay under their module.

## Reproduction — `repro/`

- [x] `run_line.py` — the P0 → P3 driver, replacing `beamline_design/
      segments.py` (and the P0 → P3 tracking the FEL-side prep script
      repeated). It
      solves NOTHING: `thzim.record` (new) holds each OP's line as data --
      section geometry, the SC-matched strengths the per-OP scripts derived,
      the P3 target -- and the driver rebuilds the three lattices from it and
      tracks section by section with `thzim.compressor.track_compressor`
      (SC everywhere, CSR in the compressor), writing every plane as ASTRA
      and a per-plane `summary.json`. `--start/--stop` re-run part of the
      line from a saved plane, `--input` runs a different beam (the 1M
      distributions) through the same line. The per-OP scripts keep their own
      settings blocks: they are where the numbers are derived, `record.py` is
      where the result is kept; a value changed in a script must be carried
      into the record by hand -- that duplication is deliberate and small.
      Not carried over from `segments.py`: the `.npz` dumps and the
      `chirp_scale` hook (unused). Everything past P3 is delivered
      separately.
- [x] `OP{1,2}/dogleg_ki_scan.py`, `OP{1,2}/dogleg_forward.py` — new; the
      two-step dogleg solve over `thzim.dogleg` (scan the free inner-pair ki,
      then run the chosen one forward). Replaces the short-lived
      `tools/dogleg_solve_demo.py`. OP2 shares the hardware AND the optics with
      OP1 — the scan reproduces OP1's table column for column — but its selected
      `ki` is the scan optimum (−22) where OP1's was hand-set (−38).
- [x] `OP{3,4}/chicane_beta_x_scan.py` — new; the collective half of the chicane
      entrance-Twiss design (y is solved analytically by `thzim.chicane`.
      `matched_y`, x is scanned with SC+CSR). The scan supplies a soft
      engineering trade-off; `beta_x=30 m`, `alpha_x=0` is the adopted nominal
      for both OP3 and OP4. OP4 is the collective-dominated point: 3x the
      charge at half the energy, so the same trade-off costs several um of
      emittance instead of one.
- [x] `OP{1,2}/dogleg_track.py`, `OP{3,4}/chicane_track.py` — new; the
      compressor stage P1 -> P2 over `thzim.compressor`, replacing `seg2` of
      `beamline_design/segments.py`. Each reads `outputs/OP<n>/beams/p1.ast`,
      rebuilds the lattice from the design of record retained in the script
      (the dogleg re-solves `ko` from `ki` exactly as the e2e script did, so
      target and lattice cannot drift apart), runs a linear reference and the
      SC + CSR track, writes `p2.ast`, and draws the evolution and the
      longitudinal phase space at P1 / P2. Deliberately not carried over from
      `segments.py`: the `.npz` beam dumps (ASTRA files via partdist instead,
      one format at every plane), the 20k subsample and the `chirp_scale` hook
      (an unused experiment).
- [x] `OP{1..4}/m3_quadruplet_match.py` — new; the final match P2 -> P3 over
      `thzim.quadruplet`, replacing `seg3` of `beamline_design/segments.py`
      for BOTH branches. OP1/OP2: the four-quad M3 after the dogleg, as
      before (d_in 0.1 / 0.3 m). OP3/OP4: **a design change.** `segments.py`
      ran the retired six-knob crossing solver over M2 + switch region + M3;
      its "round" constraint was emittance-WEIGHTED (equal beta), which on
      the P2 beam (eps_x 2-4x eps_y after the chicane) is 95-160 % from
      size-round, so `thzim.two_triplet` finds no crossing for it -- checked
      by scanning both legs over +-20 T/m with several seeds and by
      evaluating the original knobs with `match_with`. The M3-SR line is
      therefore run with T1.Q1 and T1.Q2 switched off and T1.Q3 + T2 solved
      as a quadruplet with gaps (0.92, 0.30, 0.30): linear solutions exist
      for both OPs at <= 4.5 T/m. The P3 targets are carried as constants
      (the FEL-side choice of record: (3.38, 0, 0.007, 0), (0.53, 0, 0.53,
      0), (2.58, 0, 0.07, 0), (9.8, 0, 0.006, 0)); matching is on the
      PROJECTED P2 Twiss as in `segments.py`. Each script writes
      `outputs/OP<n>/beams/p3.ast` and prints the P3 longitudinal summary
      for the FEL simulation, which is delivered separately.
- [x] Separate `knobs/` tables — deliberately dropped. They duplicated values
      without carrying enough lattice context. Geometry and selected strengths
      are kept in the executable per-OP scripts instead.

## Tools

- [x] `tools/bunch_compression_scan.py` ← merge of `bunch_compression_scan_v5.py` (FWHM axis)
      and `bunch_compression_scan_sigma_z_v1.py` (sigma_z axis, conventions/branches).
      Axis is now sigma_z,b; the colour map is labelled `sigma_pz/pz0` to match what it
      computes (was "energy spread"); `C_LIGHT = 3e8` → `scipy.constants.c`; xtils
      removed, figures saved through `plt_style.save`. Extended to a 2×2 panel per OP
      with the measured beam overlaid: the values are seeded in the settings block
      (read once from `data/OP*_50k.dist` with `measure_beam()`, the only partdist use).
- [x] `tools/bunch_compression_yield.py` — new; fixed-spread forward map
      (given R56, what peak current comes out). Shares the physics layer with
      `bunch_compression_scan.py`.
- [x] `tools/bunch_compression_r56.py` — new; transposed view of the scan
      (spread on the y axis, required R56 as the colour map). Shares the physics
      layer with `bunch_compression_scan.py`.
- [x] `tools/peak_current_map.py` — new; I_peak over (sigma_z,b, Q) for a
      chosen current profile, with the compression of each OP drawn as a
      constant-charge arrow. Shares the profile models with
      `bunch_compression_scan.py`.
- [x] `tools/chicane_geometry_map.py` — new; R56 over (theta, L_Dz) at fixed
      L_Bz, geometry from `chicane_design/no_quad/chicane_geom_scan.py`.
- [x] `tools/dogleg_geometry_map.py` — new; R56 over (theta, rho), geometry from
      `dogleg_design/dogleg_api.py: rho_for_r56()`, reduced to the closed form by
      dropping the velocity term. Both are the sizing-stage step between
      `bunch_compression_scan.py` and the lattice builders in `src/thzim/`.
- [x] `tools/chicane_offset_map.py`, `tools/dogleg_offset_map.py` — new; the
      Delta_x companions on the same axes, importing geometry from the matching
      `*_geometry_map.py`.
- [x] `tools/undulator_resonance.py` ← `demo_fel_resonance_wg_helical_3.py` (cleaned: xtils/waveguides deps removed, duplicate figure block dropped, no figure saving)
- [x] `tools/demo_quadruplet_match.py` ← the `match/four_quads_match*.py`
      family (`four_quads_match.py`, `_example.py`, `_sc_track.py`,
      `_current_scan.py`), collapsed into ONE script over `thzim.quadruplet`
      with an SC switch, on the pattern of the two-triplet demos. Same case as
      the originals (dogleg-exit Twiss (10.29, −15.4, 10.44, −6.3) onto the
      OP2 undulator waist 0.53 m) but on the real OP2 P0 beam re-conditioned by
      `partdist.match_twiss_xy` instead of a synthetic Gaussian — which is why
      `make_matched_beam` did not need migrating. Linear check on that beam:
      k1 = (+19.3, −32.9, +34.4, −100.8) 1/m², the same family as the upstream
      as-built OP2 M3 (22.8, −34.1, 34.1, −117.0) and as this package's own
      (`record.py`: +23.3, −34.1, +33.9, −126.6); peak beta 31 / 57 m. The peak-current
      scan (`_current_scan.py`, "Q4 is the sensitive knob, near-linear in I")
      is NOT reproduced: its conclusion is recorded in `thzim.quadruplet`'s
      docstring and the necessity of the SC layer is not in question.

## Docs

- [x] `docs/dogleg_design.md` — new; the SASE-branch compressor design procedure
      (sizing, achromat, entrance Twiss, ki choice), the R56 geometric-vs-tracked
      bookkeeping, and the OP1/OP2 design of record. Companion chapter for the
      chicane still to write.
- [x] `docs/overview.md` and the layout figure — dropped by decision: the
      machine overview is documented with the injector and FEL deliveries,
      not here. `docs/` holds the per-subsystem design notes only.
- [x] Root `README.md`: install/quickstart filled
- [x] `data/README.md`: define the supplied 50k P0 beams as the workflow inputs
- [x] `data/OP{1..4}_50k.dist` — canonical P0 beams, in git (`.gitignore` carries an
      explicit `!data/OP*_50k.dist` exception to the blanket `*.dist` rule)

## Deliberately NOT migrated

Derivation and scratch, per handoff decision: `backups/`, untracked scripts in
`match/` and `dogleg_design/`, root exploratory scripts (`scan_*`, `test_*`,
`plot_analytic_*`, `pg_single.py`, `batch_beam_diag.py`),
superseded pipelines
(`sr_line.py`, `sase_line.py`, `final_figures.py`, `slides_figs.py`,
`fel_design/fel_prep.py` v1, `sr_theta_fel.py`, `case*_f1p*/`), chicane study
scripts and design-principle notes (`chicane_design/no_quad/study_*.py`, `*.md`),
stale `Scripts/` templates, obsolete `data/*.dist`. The photoinjector (P0
beams and their optimisation) and the FEL stage (everything past P3) are
delivered separately: this package covers the middle optics P0 → P3 and starts
from the supplied `data/OP{1..4}_50k.dist` files.

## Companion repository (published separately)

- [ ] `partdist` — the only dependency not on PyPI (everything else --
      numpy, scipy, matplotlib, ocelot-collab -- is declared in
      `pyproject.toml`)
