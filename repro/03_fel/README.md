# Stage 3 — FEL Simulation (Genesis 1.3 v4)

Genesis4 time-dependent FEL runs for the four working points. The P3 beam from
stage 2 is converted to Genesis sliced-beam format via
`tools/partdist_astra2genesisslices`, then tracked through the undulator
(lattices in `resources/lattices/`).

## Contents

- `fel_prep.py` — build per-OP run dir: convert `p3.ast` → `beam.par.h5`,
  assemble the Genesis4 input deck (to migrate from `fel_design/fel_prep2.py`)
- `fel_run.py` — launch `mpirun genesis4` per OP
- `fel_analysis.py` — gain curves / spectra from `g4.*.out.h5`
- `OP{1..4}/` — Genesis4 input decks; large I/O (`*.h5`, `*.ast`) gitignored
- `condor/` — HTCondor submit files for farm runs
  (still to migrate from the earlier workflow)

## Requirements

`genesis4` + MPI runtime, `h5py`, `partdist`; optional `postpro` for
post-processing.

## Run

<!-- TODO: local single-OP and condor farm commands -->
