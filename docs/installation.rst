Installation
============
voxTrace is a C++20 package built with a plain Makefile. It is developed and
tested on macOS (Apple Silicon, Homebrew) and Linux (Ubuntu 20.04/22.04,
AlmaLinux on an HPC cluster). The option ``HPC=true`` stores a second set of
library paths for a cluster so the Makefile never needs editing; default is
``false``.

Dependencies
------------

Required:

* `XrayLib`_ — X-ray interaction data (the only dependency of the trace core)
* `Armadillo`_, `Ensmallen`_, `GSL`_ — used by the optimizer layer (the fit)

Optional:

* `Kokkos`_ (OpenMP or CUDA/HIP build) — for the parallel/GPU-ready binaries;
  the host-only serial build needs no Kokkos at all
* `polycap`_ — **only** for the validation test ``make test2``, which compares
  voxTrace's own polycapillary implementation against the reference library;
  the app itself never uses it

On macOS: ``brew install xraylib armadillo ensmallen gsl libomp`` (plus a
Kokkos install for the parallel build). On Ubuntu the same packages are
available via apt/pip or built from source; ``pre-install.sh`` shows one
working recipe.

Building and first runs
-----------------------

.. code-block:: bash

   make voxtrace              # host-only serial app        → build/src/voxTrace
   make voxtrace-kokkos       # Kokkos (OpenMP/CUDA) app    → build/src/voxTraceK
   make test2 test4           # validation: polycap reference, XRR physics

   # beam characterisation: source + primary optic only
   ./build/src/voxTrace test-data/simulation/beam-pc236

   # full confocal chain + voxel-weight reconstruction
   ./build/src/voxTrace test-data/simulation/nist-1107-recon

   # any Setup.txt key can be overridden on the command line
   ./build/src/voxTrace test-data/simulation/nist-1107-recon \
        chain=source,primary,sample,detector variance_reduction=0 fit=0

A simulation directory (``Polycapillary.txt``, ``Source.txt``,
``Capillaries.txt``, ``Sample.txt``, ``Materials.txt``, ``Simulation.txt``,
``Setup.txt``) fully describes an experiment — copy one of the examples under
``test-data/simulation/`` and modify it to your needs. For a CUDA build,
install Kokkos with the CUDA backend and build with
``make voxtrace-kokkos KOKKOS_INSTALL=<prefix> KOKKOS_CXX=<prefix>/bin/nvcc_wrapper``.

Building the documentation
--------------------------
This site is generated from the in-source Doxygen comments and the ``.rst``
pages in ``docs/`` by Doxygen + Sphinx/Breathe. To build it locally:

* Install ``doxygen`` (e.g. ``apt install doxygen`` / ``brew install doxygen``).
* Install the Python dependencies: ``pip install -r docs/requirements.txt``
  (ideally inside a virtual environment).
* Run ``make docs`` — the HTML site is written to ``build/doc/html/index.html``.

A short statement by the author, concerning licencing:
-------------------------------------------------------
This code is supposed to be free to use, without any warranty from my side.
I therefore chose the `MIT Licence`_. Nevertheless, before
using/redistributing this code in a commercial way you should notice that
some of the packages this code depends on have different licences:

* `XrayLib`_: a library for interactions of X-rays with matter
* `Armadillo`_: a C++ library for linear algebra & scientific computing
* `Ensmallen`_: a flexible C++ library for efficient numerical optimization
* `GSL`_ — GNU Scientific Library (gradient-free optimization algorithms)
* `Kokkos`_ (optional): performance-portable parallel programming model
* `polycap`_ (optional, validation only): X-ray transmission through
  polycapillaries

.. _Armadillo: https://arma.sourceforge.net/
.. _Ensmallen: https://ensmallen.org/
.. _GSL: https://www.gnu.org/software/gsl/
.. _XrayLib: https://github.com/tschoonj/xraylib/wiki
.. _polycap: https://github.com/PieterTack/polycap
.. _Kokkos: https://kokkos.org/
.. _MIT Licence: https://michaeliro.github.io/voxTrace/licence.html
