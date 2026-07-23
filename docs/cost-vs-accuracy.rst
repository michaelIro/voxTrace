Computational cost vs. physical accuracy
========================================

Monte-Carlo tracing lets you choose where to sit between two extremes: a
*physically literal* simulation, in which every photon does exactly what a
real photon would do, and an *importance-sampled* one, which spends its
computing time only on photons that can ever be detected. voxTrace implements
both ends behind one switch in ``Setup.txt``::

   variance_reduction = 1     # importance sampling (default, ScanKernel)
   variance_reduction = 0     # brute force: literal analog transport (AnalogKernel)

Both run on every backend — serial, OpenMP, and the CUDA-ready Kokkos build —
and both are reproducible bit-for-bit at any thread count thanks to per-ray
RNG streams. What differs is what one detected photon *costs* and what the
result *means*. Numbers below are measured on the ``nist-1107-recon`` example
(brass, PC-236 optics, 17.4 keV).

Brute force: the physically literal reference
---------------------------------------------

With ``variance_reduction = 0`` each GPU/CPU thread owns one slot of the
requested number of *detected* photons per scan position (``Simulation.txt``'s
rays-per-measurement-point, or the ``n_detected`` key). The thread draws a
candidate photon and follows it literally; if the photon dies anywhere, the
thread **respawns the next candidate from the source** — a while loop that
only ends when its slot holds a detected photon (or ``max_attempts`` is
exhausted). Along the way nothing is weighted:

* the free flight through the voxel grid *is* the attenuation — no
  ``exp(-τ)`` factors, the photon either reaches the next interaction or
  leaves the sample;
* at each interaction the photon is photo-absorbed (re-emerging isotropically
  as a fluorescence photon, or dying by Auger), or Rayleigh/Compton-scattered
  into a new direction drawn from the tabulated θ distributions — and the new
  photon **continues, to any order**: scatter → excitation → fluorescence →
  scatter → detector chains happen exactly as in the real sample (capped by
  ``max_generations``);
* the secondary optic's transmission and the detector efficiency are survival
  *draws*, not weights.

Every recorded count is therefore one photon, with true Poisson statistics
and no model approximations. In the brass example roughly **7 % of detected
photons are second- or higher-order** (measured from the generation
histogram: 88 000 first-order, 6 200 second, 300 third, 18 fourth per
~95 000 interactions) — this inter-element enhancement is physics that only
this mode contains.

The price is the detection probability. A candidate photon must survive the
beam-transmission draw, interact, scatter or fluoresce *by chance* into the
collection solid angle, and survive optic and detector draws:

===========================================  ======================  ======================
chain                                        p(detect) / candidate   attempts per photon
===========================================  ======================  ======================
bare detector aperture (no secondary optic)  ~1.6 × 10⁻³ (measured)  ~6 × 10²
full confocal (secondary polycapillary)      ~10⁻⁵ … 10⁻⁶            10⁵ – 10⁶
===========================================  ======================  ======================

**Consider the hardware.** An 8-core laptop pushes ≈ 6 × 10⁶ candidates per
second. The aperture chain (500 photons × 11 positions ≈ 7 × 10⁶ candidates)
finishes in about a second — brute force is perfectly usable there. The full
confocal chain at survey statistics (30 000 photons × 11 positions ×
~10⁵ attempts ≈ 3 × 10¹⁰ candidates) is an **hours-long CPU job**; that is
the workload the CUDA Kokkos build exists for, where a data-center GPU is
expected to shorten it by one to two orders of magnitude. ``max_attempts``
guards against a chain whose detector is effectively unreachable — slots that
hit the cap are reported, not silently dropped.

Variance reduction: the fast, differentiable estimate
------------------------------------------------------

With ``variance_reduction = 1`` every trick keeps the *expected value* honest
while removing the waste. In the order a photon meets them:

1. **Shared beam.** The primary optic is traced once; every scan position
   reuses the exit rays. Identical physics, pure speed.
2. **Aimed emission.** Instead of emitting isotropically and hoping, the
   emitted photon is aimed at a random point of the collection window and
   carries the solid-angle probability as a weight. This alone recovers the
   factor ΔΩ/4π ≈ 10³–10⁶ that brute force pays, and is *exact* — for the
   first interaction order.
3. **Weights instead of survival draws.** Self-absorption ``exp(-τ)``, the
   secondary optic's transmission and the detector efficiency multiply the
   photon's weight instead of killing it. Zero variance is added by these
   steps; in exchange, counts become weighted estimates, so a spectrum's
   noise is *not* purely Poisson (the app's ``expo_per_event`` calibration
   and the χ²-consistency notes it prints account for this).
4. **Single-interaction linearisation.** The one real *physics* trade: the
   emitted photon never interacts again, so enhancement and multiple
   scattering are absent — expect line-ratio biases at the level of the
   higher-order fraction (order of a few per cent in brass, more in strongly
   enhancing systems). This is also what makes the detected spectrum
   **linear** in the per-voxel weights, ``S(w) = B + Σ wᵥRᵥ``, which is what
   gives the reconstruction its response matrix, its exact analytic gradient
   and its fast L-BFGS fit. Full-order analog data has no such linear
   structure.
5. **Polarization.** The azimuthal scatter modulation is applied as a weight
   in this mode only — the tabulated analog θ distributions are unpolarized.

Choosing a mode
---------------

=====================================================  =================  =========================
task                                                   mode               hardware
=====================================================  =================  =========================
reconstruction fits, parameter surveys, quick looks    ``vr = 1``         laptop is fine
beam characterisation (no sample)                      either             laptop is fine
absolute reference spectra, Poisson-true statistics    ``vr = 0``         GPU / HPC (or overnight)
quantifying enhancement / validating the vr-bias       ``vr = 0``         GPU / HPC
=====================================================  =================  =========================

A good workflow is hybrid: run the fit with variance reduction, then spend
one brute-force run at the fitted sample to measure what the linearisation
missed.

Configuration keys
------------------

All in ``Setup.txt`` (or as ``key=value`` command-line overrides):

* ``variance_reduction`` — the switch described above.
* ``n_detected`` — brute force: detected photons per scan position
  (0 → the rays-per-point value in ``Simulation.txt``).
* ``max_attempts`` — brute force: respawn cap per detected-photon slot.
* ``max_generations`` — brute force: interaction-order cap per photon.
* ``expo_per_event`` — variance reduction: measured counts per detected
  Monte-Carlo event in the synthetic measurement (balances Poisson vs
  Monte-Carlo model noise).
* ``polarization`` — scatter polarization weights (variance reduction only).
