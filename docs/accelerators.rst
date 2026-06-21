Accelerators & performance portability
======================================

voxTrace began as a CUDA-only code and was ported to a *performance-portable*
model: one header-only core that runs on multi-core CPUs, NVIDIA and AMD GPUs,
and Apple-silicon GPUs without source changes. This is achieved with two
ingredients — `Kokkos`_ for CPU/NVIDIA/AMD and a Metal path for Apple — behind a
single thin abstraction header, ``src/core/Platform.hpp``.

The Platform.hpp abstraction
----------------------------

Every device-callable function is annotated with portable macros that expand
differently per target. ``Platform.hpp`` defines them for three regimes:

============================  ====================================  ==================================
Macro                          Kokkos (CPU/CUDA/HIP)                 Metal (MSL)
============================  ====================================  ==================================
``KOKKOS_INLINE_FUNCTION``     ``Kokkos``'s host/device inline        ``inline``
``VT_SCONSTEXPR``              ``static constexpr``                   ``static constant constexpr``
``VT_DEVICE_METH`` / ...       *(empty)*                              ``device`` address-space qualifier
============================  ====================================  ==================================

Selection is by predefined macros:

* ``__METAL_VERSION__`` — set by the Metal compiler → MSL path.
* ``VOXTRACE_HOST_ONLY`` / ``VOXTRACE_METAL`` — plain host build, no Kokkos
  headers (used by the host-only tests, e.g. ``make test2``).
* otherwise — full Kokkos build (``#include <Kokkos_Core.hpp>``).

Because the macros are the only backend-specific tokens in the physics code, a
class like :cpp:class:`Ray` or :cpp:class:`Voxel` is written once and is valid in
all three regimes.

Kokkos backends (CPU / NVIDIA / AMD)
------------------------------------

The dispatch layer (:cpp:func:`Tracer::callTraceNewBeam`,
:cpp:func:`PolyCap::traceBatch`) uses ``Kokkos::parallel_for`` /
``parallel_reduce`` over the ray index, which Kokkos maps to:

* **Serial / OpenMP** on CPUs — the default; selected via the ``KOKKOS_INSTALL``
  build with the corresponding device enabled.
* **CUDA** on NVIDIA GPUs — build with
  ``make KOKKOS_INSTALL=... KOKKOS_CXX=$(HOME)/kokkos/bin/nvcc_wrapper``.
* **HIP** on AMD GPUs — analogous, with a HIP-enabled Kokkos install.

Data lives in ``Kokkos::View`` buffers (rays, voxels, materials, elements) that
are mirrored host↔device once around the kernel; the trace itself does no host
communication.

Metal (Apple silicon)
---------------------

On arm64 macOS the Makefile additionally compiles ``src/metal/Tracer.metal`` to
a ``.metallib`` and links a small Objective-C++ driver
(``src/metal/MetalTracer.mm``). The MSL kernel ``#include``\s the same core
headers; the ``__METAL_VERSION__`` branch of ``Platform.hpp`` makes them valid
MSL. This is why the core is restricted to ``float`` and fixed-size arrays
(:doc:`architecture`): MSL supports neither ``double`` nor heap allocation.

.. note::

   ``PolyCap`` keeps its transient per-photon trace state in ``double`` for
   off-axis precision and therefore does **not** compile in a Metal shader as-is.
   A future all-float reformulation (tracking the photon as an offset from the
   capillary axis) would restore Metal compatibility — see ``TODO.md`` and the
   precision note in ``PolyCap.hpp``.

Building for each target
------------------------

.. code-block:: bash

    # CPU (Serial/OpenMP) — default
    make KOKKOS_INSTALL=$HOME/kokkos-install

    # NVIDIA CUDA
    make KOKKOS_INSTALL=$HOME/kokkos-cuda \
         KOKKOS_CXX=$HOME/kokkos/bin/nvcc_wrapper

    # Apple Metal is detected automatically on arm64 when Xcode is present.

    # Host-only (no Kokkos) — used by the validation tests
    make test2

Profiling each backend is tracked in ``TODO.md`` (Kokkos Tools, Nsight,
``perf``).

.. _Kokkos: https://kokkos.org/
