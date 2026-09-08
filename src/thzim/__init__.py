"""thzim — THz FEL ideal machine design toolchain.

Core modules:

- ``thzim.chicane``  — chicane geometry, lattice building, SC/CSR tracking
- ``thzim.dogleg``   — achromatic dogleg design
- ``thzim.triplet``  — round-transport triplet solver (the matching building block)
- ``thzim.maps``     — single-plane 2x2 linear transfer maps (numpy-only)
- ``thzim.utils``    — figure style, and the beam-rigidity helpers
- ``thzim.compressor`` — SC + CSR tracking through either compressor, with
  the along-s beam monitor and the exit-plane report

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
internally and do not expose the ocelot object.
"""

__version__ = "0.1.0"
