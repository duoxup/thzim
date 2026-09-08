# thzim — THz FEL Ideal Machine

Simulation design code for a THz FEL ideal machine: start-to-end workflow from
the supplied P0 particle distributions through middle-optics beam transport
(Ocelot) to FEL simulation (Genesis 1.3 v4).

> **Status:** the importable optics package, canonical P0 inputs, subsystem
> design studies, and per-OP P0 → P1 matching scripts are in place. Remaining
> integration work is tracked in [MIGRATION.md](MIGRATION.md).

## Repository map

| Path | Category | Content |
|---|---|---|
| `src/thzim/` | Toolchain | Importable library: chicane, compressor, dogleg, maps, quadruplet, triplet, two_triplet, utils |
| `repro/02_beamline/` | Reproduction | Middle optics P0 → P3 design and tracking per working point |
| `repro/03_fel/` | Reproduction | Genesis4 FEL runs for OP1–OP4 (local + HTCondor) |
| `resources/` | Static inputs | Genesis undulator lattices |
| `tools/` | Toolchain | Command-line utilities (ASTRA → Genesis conversion) |
| `data/` | Data | Canonical 50k-particle P0 distributions for OP1–OP4 |
| `docs/` | Docs | Machine overview; per-subsystem design notes |

## Working points

Four permanent physics working points are used throughout (`OP1`–`OP4`):
SASE branch (OP1: 1 THz, OP2: 10 THz) and superradiant branch
(OP3: 1 THz, OP4: 0.3 THz). See `docs/overview.md`.

## Install

<!-- TODO: verify once code lands -->

```bash
pip install -e .
```

Companion packages (separate repositories):

- `partdist`
- `htpipe`
- `paramstudy`
- `postpro`

External simulation code: Genesis 1.3 v4 + MPI (FEL).

## Data

The canonical P0 beams `data/OP{1..4}_50k.dist` (50k particles, about 5.2 MB
each) ship with the repository and are the starting point of its simulation
workflow. See [data/README.md](data/README.md).

## Outputs

Generated files are written under `outputs/` by default: tool figures go to
`outputs/tools/figures/`, reproduction figures to
`outputs/OP<n>/figures/`, and tracked segment distributions to
`outputs/OP<n>/beams/`. Set `THZIM_OUTPUT_DIR` to use a different output root.
Missing output directories are created automatically; `outputs/` is ignored by
Git.

## Quickstart

<!-- TODO: per-stage run commands once code lands; see repro/*/README.md -->
