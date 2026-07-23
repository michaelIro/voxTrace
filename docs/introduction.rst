What is voxTrace?
===================
voxTrace is a voxel-based Monte-Carlo ray-tracing C++ code for quantitative
(confocal) micro X-ray fluorescence analysis. Its main tested purpose is the
simulation — and iterative quantitative *reconstruction* — of energy-dispersive
micro-XRF spectra in a confocal setup (CMXRF).

One executable, ``voxTrace``, runs a configurable instrument. A simulation
directory of plain text files describes the beamline, and its ``chain`` line
decides which stages are mounted::

   chain = source,primary,sample,secondary,detector

* **source** — the X-ray source (monochromatic or a sampled spectrum),
* **primary** — a polycapillary full lens focusing the beam into the sample
  (voxTrace's own single-header implementation, validated against the
  reference `polycap`_ library),
* **sample** — the voxel grid: per-voxel materials, attenuation,
  photoelectric absorption/fluorescence, Rayleigh and Compton scattering,
* **secondary** — a second polycapillary selecting the confocal volume
  (without it, a bare detector aperture collects — classic scanning µXRF),
* **detector** — the Si(Li)/Si-PIN response: efficiency, escape peaks,
  Compton continuum, finite resolution.

Chains without a sample characterise the beam after the optic; chains with
sample and detector simulate spectra at every scan position and can *fit* the
per-voxel sample weights so that simulated spectra match measured ones
(loaded from file, or a synthetic phantom) by minimising a χ² or
ROI-weighted χ² with ensmallen's L-BFGS.

Two Monte-Carlo modes cover the accuracy/cost trade-off (see
:doc:`cost-vs-accuracy`): importance sampling for fast, differentiable
estimates, and a brute-force analog mode in which every photon is transported
literally — to any interaction order — until it is detected or lost.

The physics core is header-only and *performance portable*: the same source
compiles serially, for multi-core CPUs (OpenMP) and for NVIDIA/AMD GPUs via
`Kokkos`_, with bit-identical results at any thread count. The code
originated as a CUDA-only project; the :doc:`architecture` and
:doc:`accelerators` pages describe the portable design that replaced the
CUDA-specific layer.

.. _polycap: https://pietertack.github.io/polycap
.. _Kokkos: https://kokkos.org/
