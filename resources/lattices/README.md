# Genesis4 Undulator Lattices

Single source for the per-OP undulator lattice files `OP{1..4}.lat`
(APPLE-II helical). FEL run directories reference these instead of keeping
their own copies.

| OP | λu | periods in file | design output at |
|---|---|---|---|
| OP1, OP2 (SASE) | 65 mm | 120 | 60 periods |
| OP3, OP4 (SR) | 130 mm | 10 | 5 periods |

The files are written at **double** the design length so the gain curve can be
read past the design point; mind this when analysing `g4.*.out.h5`.

The four lattice files are still to be added; see MIGRATION.md.
