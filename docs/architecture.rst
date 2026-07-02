Code architecture
=================

voxTrace is a data-oriented Monte-Carlo ray tracer. The whole physics core is
header-only and is written so that the *same* source compiles for a CPU and for
an NVIDIA/AMD GPU through `Kokkos`_. This page explains the design that makes
that possible; for the build-time mechanics of each backend see
:doc:`accelerators`.

The ray is the unit of parallelism
----------------------------------

Every photon is a :cpp:class:`Ray` value. A simulation is "trace N independent
rays", so the ray is the natural thread / parallelisation anchor: the dispatch
layer launches one thread per ray and each thread owns its ``Ray`` for the
whole trace. Nothing about a ray is shared, so there is no synchronisation in
the hot loop.

All the other core classes are *stateless operators* that take a ``Ray&`` and
mutate it in place:

.. code-block:: text

    Source*     -> creates a Ray (position, direction, energy, polarisation)
                   *SourceBase hierarchy: Source (mono) / XRayTube /
                    Synchrotron / LiquidMetalJet
    PolyCap     -> trace(Ray&)          : push the ray through the optic
    Sample      -> findStartVoxelIdx    : locate the entry voxel
    Voxel       -> intersect(Ray&)      : path length + next-voxel index
    Material    -> getInteractingElementIdx
    ChemElement -> getInteractionType / getThetaCompt / ...  : the physics
                   that changes the ray's energy and direction
    Detector    -> detect(E, ...)       : Si(Li) response — efficiency,
                   escape peaks, Compton continuum, finite resolution

The delegation chain mirrors the physical hierarchy: a ray hits the **sample**,
the sample finds which **voxel**, the voxel knows its **material**, the material
picks the interacting **element**, and the element samples the actual
scattering/fluorescence event that rewrites the ray. Control then returns to the
loop, which advances the ray to the next voxel and repeats. See
:cpp:func:`Tracer::traceForward` for the single-ray step and
:doc:`classes` for the per-class API.

Design rules (why the classes look the way they do)
---------------------------------------------------

These constraints come from the GPU target and are applied uniformly:

* **Value types, no virtual calls.** Runtime polymorphism on the GPU is
  expensive. Behaviour is selected by data (e.g.
  :cpp:func:`ChemElement::getInteractionType`) rather than by vtables. Where a
  real class hierarchy is useful — the source models — it is expressed with the
  *static* Curiously-Recurring-Template-Pattern (:cpp:class:`SourceBase`), so the
  dispatch is resolved at compile time, there is no vtable, and the types stay
  trivially copyable into device memory.
* **Fixed-size arrays, no dynamic allocation in device code.** GPU kernels can
  not heap-allocate. Per-element tables such as ``ChemElement::dcs_rayl[100][628]``
  are fixed-extent members; the trace never calls ``new``.
* **Shared data is passed at the call site, not owned.** Operators receive the
  shared ``ChemElement``/``Material`` arrays as parameters
  (``const ChemElement* elems``) instead of holding pointers, so the structs
  stay trivially copyable into device memory.
* **Heavy host work is done once, at construction.** Anything that needs an
  external library (xraylib) runs in a host-only constructor that fills the
  fixed tables — see :cpp:class:`ChemElement` and :cpp:class:`PolyCap`. Device
  code only ever *reads* those tables, so the trace has no runtime dependency.
* **float by default.** 32-bit halves memory traffic and is faster on GPUs, so
  all stored state is ``float``. The one exception is the transient ``PolyCap``
  trace state, which must be ``double`` for off-axis grazing-angle precision (it
  never touches the GPU storage layout; see the precision note in
  ``PolyCap.hpp``).

The two executables
-------------------

* **SampleTracer** — generates the primary beam, traces it through the sample
  voxel grid, and records the X-ray/matter interactions.
* **CapillaryTracer** — traces the post-sample photons through the secondary
  polycapillary optic to the detector.

Both are thin drivers around the same header-only core; only the dispatch and
I/O differ. See :doc:`use` for the input files they consume.

.. _Kokkos: https://kokkos.org/
