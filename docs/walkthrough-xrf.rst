Trace walkthrough — XRF (bare source)
=====================================

The XRF preset is the shortest chain:

    **source → sample → detector**

.. code-block:: text

   chain = source,sample,detector

No optics at all: the source illuminates the sample directly and a bare
detector aperture collects the fluorescence. It reuses the µXRF collection
side (:doc:`walkthrough-muxrf`) and simplifies the beam side; as always, the
full step-by-step lives in :doc:`walkthrough-cmxrf`.

The beam without an optic
-------------------------

``BeamKernel`` still runs — one thread per primary photon, each with its own
dice and (for spectrum sources) its own dice-drawn energy — but with
``hasOptic = false`` it short-circuits after birth:

.. code-block:: cpp

   double e = sampleEnergy(rng);                    // spectrum dice still apply
   double rr = sqrtf(rng.frand()) * srcR, ra = TWO_PI_D * rng.frand();
   Vec3 p0 = {rr*cos(ra), rr*sin(ra), 0};
   if (!hasOptic) {
       slots[i] = {p0, {0, 0, 1}, 1.0, e};          // parallel ray, unit weight
       return;
   }

Every photon survives: a uniform disk of parallel rays with weight 1 — an
idealized flood beam whose radius is the source radius from ``Source.txt``.
There is no transmission loss and no focusing, so:

* **nothing is thrown away in stage 1** — ``n_primary`` photons in, the same
  number of beam rays out (memory note: the surviving-beam array is now as
  large as the request, so very large ``n_primary`` values cost host RAM;
  see the scaling table in the :doc:`hardware-primer`);
* the "focus" is meaningless — the beam illuminates a source-radius-sized
  footprint on the sample, and the kernel aims the sample frame one
  centimetre of standoff behind the exit plane
  (``zFocPrim = 1.0`` in ``voxTrace.cpp``: *"bare beam: 1 cm standoff"*).

The sample and the detector
---------------------------

From surface entry onward the trace is exactly the µXRF one:

1. transform into the sample frame, intersect the surface, discard misses;
2. voxel walk to the interaction point (Beer–Lambert dice);
3. emission aimed at the bare detector aperture (importance sampling) or
   isotropic with a geometric disk test (analog);
4. fluorescence-or-scatter dice — Auger losers are discarded, Rayleigh and
   Compton feed the background;
5. self-absorption ``exp(−τ)`` on the way out;
6. *(no optic step — ``wSec = 1``)*;
7. Si(Li) detector response: efficiency, escape peak, resolution broadening,
   channel binning → one ``Event`` per surviving photon.

What that means physically
--------------------------

This is classic bulk XRF: a wide beam, no lateral or depth selectivity, the
whole illuminated volume contributes. It is the configuration to use as a
sanity baseline — count rates are highest, geometry effects are smallest, and
a homogeneous sample should reproduce tabulated fluorescence intensity ratios
directly. Comparing the same sample across the three walkthroughs shows
exactly what each optic buys:

===================  ==========================  =================================
Preset               Beam side                   Collection side
===================  ==========================  =================================
XRF                  flood beam, no losses        bare aperture, whole column
µXRF                 focused spot (~10 µm)        bare aperture, whole column
CMXRF                focused spot (~10 µm)        confocal optic, probe volume only
===================  ==========================  =================================

.. admonition:: Where this runs

   The beam kernel is now trivial (microseconds); essentially the entire run
   is the scan kernel — still one photon per thread on whatever backend the
   build selected (OpenMP threads on CPU cores, CUDA threads on an NVIDIA
   GPU). Because no photons die in stage 1, thread utilisation in stage 2 is
   at its highest of the three presets.
