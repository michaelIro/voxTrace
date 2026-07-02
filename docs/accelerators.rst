Accelerators & performance portability
======================================

voxTrace began as a CUDA-only code and was ported to a *performance-portable*
model: one header-only core that runs on multi-core CPUs and on NVIDIA and AMD
GPUs without source changes, through `Kokkos`_, behind a single thin abstraction
header, ``src/core/Platform.hpp``.

The Platform.hpp abstraction
----------------------------

Every device-callable function is annotated with portable macros that expand
differently per target. ``Platform.hpp`` defines them for two regimes:

============================  ====================================  ==================================
Macro                          Kokkos (CPU/CUDA/HIP)                 Host-only
============================  ====================================  ==================================
``KOKKOS_INLINE_FUNCTION``     ``Kokkos``'s host/device inline        ``inline``
``VT_SCONSTEXPR``              ``static constexpr``                   ``static constexpr``
``VT_DEVICE_METH`` / ...       *(empty)*                              *(empty)*
============================  ====================================  ==================================

Selection is by a predefined macro:

* ``VOXTRACE_HOST_ONLY`` — plain host build, no Kokkos headers (used by the
  standalone tests, e.g. ``make test2`` / ``test3`` / ``test4``).
* otherwise — full Kokkos build (``#include <Kokkos_Core.hpp>``).

The ``VT_*`` macros are no-ops today (they carried address-space qualifiers for
the retired Metal backend); they are kept as the single place a future backend
would redefine, so the physics classes stay backend-agnostic. Because these
macros are the only backend-specific tokens in the physics code, a class like
:cpp:class:`Ray` or :cpp:class:`Voxel` is written once and is valid in both
regimes.

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

.. note::

   An Apple-silicon (Metal/MSL) backend existed early on but was removed: it had
   drifted out of sync with the core and Metal cannot compile ``double`` (which
   ``PolyCap`` requires for off-axis grazing-angle precision). GPU support is now
   Kokkos-only; a Metal path could return either through a Kokkos Metal backend
   or a fresh MSL kernel, but only after an all-float ``PolyCap`` reformulation
   (see ``TODO.md``).

Building for each target
------------------------

.. code-block:: bash

    # CPU (Serial/OpenMP) — default
    make KOKKOS_INSTALL=$HOME/kokkos-install

    # NVIDIA CUDA
    make KOKKOS_INSTALL=$HOME/kokkos-cuda \
         KOKKOS_CXX=$HOME/kokkos/bin/nvcc_wrapper

    # Host-only (no Kokkos) — used by the validation tests
    make test2

Profiling each backend is tracked in ``TODO.md`` (Kokkos Tools, Nsight,
``perf``).

.. _Kokkos: https://kokkos.org/
