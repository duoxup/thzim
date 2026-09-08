# Data

Canonical P0 beam distributions for the four operating points. P0 is the
booster2 exit. These 50k-particle ASTRA-format files are the starting point of
the simulation workflow in this repository; their upstream generation and
downsampling are outside its scope.

## Operating points

| OP | Branch | Frequency | Charge | E_kin | Compressor |
|---|---|---|---|---|---|
| OP1 | SASE | 1 THz | 1.0 nC | 15.4 MeV | dogleg |
| OP2 | SASE | 10 THz | 1.0 nC | 39.4 MeV | dogleg |
| OP3 | superradiant | 1 THz | 0.2 nC | 39.9 MeV | chicane |
| OP4 | superradiant | 0.3 THz | 0.6 nC | 21.8 MeV | chicane |

## Input beams — `OP{1..4}_50k.dist`

Each file contains 50k macroparticles and is about 5.2 MB. The files are
tracked in git and used directly by the beamline design, matching and tracking
scripts. Comparisons between working points should use these same canonical
inputs so that particle statistics are consistent.

## Naming note

These were called `case1`–`case4` in the original research repository
(`py4pitz`); `OP<n>` maps one-to-one onto the old `case<n>`.
