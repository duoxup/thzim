"""thzim — THz FEL ideal machine design toolchain.

Core modules:

- ``thzim.chicane``  — chicane geometry, lattice building, matched vertical Twiss
- ``thzim.dogleg``   — achromatic dogleg design
- ``thzim.triplet``  — round-transport triplet solver (the matching building block)
- ``thzim.maps``     — single-plane 2x2 linear transfer maps (numpy-only)
- ``thzim.utils``    — figure style, and the beam-rigidity helpers
- ``thzim.compressor`` — SC + CSR tracking through either compressor, with
  the along-s beam monitor and the exit-plane report
- ``thzim.record``   — the design of record per operating point, as data:
  section geometry, selected strengths, P3 target; what ``repro/
  run_line.py`` rebuilds the line from

Two modules match a beam onto a target Twiss, by different means:

- ``thzim.quadruplet``  — four quads against the four exit Twiss numbers,
  solved on the transfer matrix and then corrected under space charge. The
  MODEL route: its residual is a Twiss the machine cannot read directly.
- ``thzim.two_triplet`` — two round-transported triplets joined where their
  screen-size curves cross. The OPERATIONAL route: it matches on beam sizes
  measured in the section between them.


Conventions
-----------
Any function that takes a particle bunch takes a **partdist**
``ParticleDistribution3D``, and only that -- one format at every entry point,
so support for a new one is added in partdist rather than as a branch in every
signature. Conversion happens at the call site::

    from partdist import from_ocelot_particle_array, to_ocelot_particle_array

Modules that must hand the bunch to ocelot (space-charge tracking) convert
internally, with ``thzim.triplet.as_ocelot``; where a script needs the exit
bunch (to write a plane or to feed the next section) the tracking functions
return the ocelot ``ParticleArray`` and the script converts back with
``partdist.from_ocelot_particle_array``. ``thzim.compressor.beam_report`` is the
one function that takes a ``ParticleArray`` directly, so an exit bunch can be
summarised without a round trip.

What this package exports at the top level is the submodules plus the few
cross-cutting names every script uses; the geometry classes, ``build_lattice``,
``element_spans`` and ``bmag`` exist in several modules and stay under them.
"""

from thzim import (chicane, compressor, dogleg, maps, quadruplet, record,  # noqa: F401
                   triplet, two_triplet, utils)
from thzim.compressor import beam_report, track_compressor
from thzim.record import RECORDS
from thzim.record import get as get_record
from thzim.triplet import as_ocelot, beam_energy_gev
from thzim.utils import M_E_GEV, apply_style, brho, output_dir, save

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "chicane", "compressor", "dogleg", "maps", "quadruplet", "record",
    "triplet", "two_triplet", "utils",
    "M_E_GEV", "brho", "apply_style", "output_dir", "save",
    "as_ocelot", "beam_energy_gev", "track_compressor", "beam_report",
    "RECORDS", "get_record",
]
