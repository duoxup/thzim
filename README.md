# thzim — THz FEL Ideal Machine

Design code for the middle optics of a THz FEL ideal machine: from the
supplied P0 particle distributions (booster exit) through matching and bunch
compression (Ocelot, space charge + CSR) to the undulator entrance P3. The
injector upstream of P0 and the FEL simulation downstream of P3 are delivered
separately.

> **Status:** the importable optics package, the canonical P0 inputs, the
> per-OP design scripts for all three sections (M1, compressor, M3) and the
> P0 → P3 driver are in place for OP1–OP4. What remains (overview document,
> a few library clean-ups) is tracked in [MIGRATION.md](MIGRATION.md).

## Repository map

| Path | Category | Content |
|---|---|---|
| `src/thzim/` | Toolchain | Importable library: chicane, compressor, dogleg, maps, quadruplet, record, triplet, two_triplet, utils |
| `repro/` | Reproduction | Middle optics P0 → P3 design and tracking per working point |
| `tools/` | Toolchain | Sizing maps, standalone calculators and demos |
| `data/` | Data | Canonical 50k-particle P0 distributions for OP1–OP4 |
| `docs/` | Docs | Machine overview; per-subsystem design notes |

## Working points

Four permanent physics working points are used throughout (`OP1`–`OP4`):
SASE branch (OP1: 1 THz, OP2: 10 THz) and superradiant branch
(OP3: 1 THz, OP4: 0.3 THz). See `docs/overview.md`.

## Install

```bash
pip install -e .
```

Companion package (separate repository): `partdist`, the particle
distribution I/O and manipulation library every script here reads beams with.

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

```bash
pip install -e .                                  # plus partdist, see above
python repro/run_line.py OP3          # P0 -> P3 for one working point
```

`run_line.py` rebuilds the line from the design of record (`src/thzim/record.py`)
and writes the P1/P2/P3 beams and a summary under `outputs/OP3/line/`. The
per-OP scripts in `repro/` are where each design value is derived;
see that directory's README for the order. The P3 beams are the handover to
the FEL simulation, which is delivered separately.
