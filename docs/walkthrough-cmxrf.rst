Trace walkthrough — CMXRF (confocal)
====================================

This page follows one simulation of the full confocal chain

    **source → primary polycapillary → sample → secondary polycapillary → detector**

from the first line of ``main()`` to the finished spectra, showing the actual
code at every step. It is written for readers who do not program — every
snippet is followed by a plain-language account of what it does. Read the
:doc:`hardware-primer` first for what *host*, *device*, *kernel* and *thread*
mean here.

The cast: what gets built from your input files
-----------------------------------------------

Everything starts in ``src/apps/voxTrace.cpp``. ``main()`` calls
``loadConfig()``, which turns the plain-text files of a simulation directory
into C++ objects:

.. code-block:: cpp

   C.run = vtio::loadSetup(C.dir + "/Setup.txt");          // chain, statistics, switches
   C.so  = vtio::loadSource(C.dir + "/Source.txt");        // source geometry + energy grid
   C.bm  = vtio::loadPlacement(C.dir + "/Placement.txt");  // bench positions and tilt angles
   C.pcPrim = vtio::loadPolyCap(dir + "/Primary_Polycapillary.txt");
   C.pcSec  = vtio::loadPolyCap(dir + "/Secondary_Polycapillary.txt");
   C.sd  = vtio::loadSample(C.dir);                        // Sample.txt + Materials.txt
   C.sc  = vtio::loadScan(C.dir + "/Simulation.txt");      // the scan positions

``Setup.txt`` is the run recipe — most importantly ``chain =
source,primary,sample,secondary,detector``, which is exactly the CMXRF preset.
From these descriptions ``main()`` constructs the working objects:

* two ``PolyCap`` optics — ``C.pcPrim.build()`` and, because the collecting
  optic is the same physical device mounted the other way round,
  ``C.pcSec.reversed().build()``;
* the **sample stage**: a voxel grid (one ``Voxel`` per cell with a
  precomputed 27-neighbour table), the list of distinct ``Material``
  compositions, and one ``ChemElement`` per element with its cross-section
  tables — all uploaded to the device once via ``DeviceBuffer``;
* a ``Detector`` — seven numbers describing a Si(Li) detector (crystal
  thickness, Be window, dead layer, Fano factor, electronic noise);
* the **beam energy**: the maximum of the ``Source.txt`` energy grid, or —
  if the grid carries a second array of per-energy weights — a full sampled
  spectrum (see stage 1).

.. admonition:: Where this runs

   All of the above is **host** code — file parsing and object building take
   milliseconds. The two expensive parts that follow, the *beam trace* and the
   *scan*, are device kernels.

Stage 1 — the beam: source through the primary optic
----------------------------------------------------

The beam is traced **once** and reused for every scan position — the photons
arriving at the sample don't care where the sample is. ``traceBeam()`` fires
``n_primary`` photons in chunks of 2 million; each photon index becomes one
thread running ``BeamKernel``:

.. code-block:: cpp

   KOKKOS_INLINE_FUNCTION void operator()(long i) const {
       slots[i].w = -1.0;                              // presumed dead
       RNG rng = makeRng(seed, (uint64_t)(base + i));  // this photon's own dice
       double e = sampleEnergy(rng);                   // its energy
       double rr = sqrtf(rng.frand()) * srcR, ra = TWO_PI_D * rng.frand();
       Vec3 p0 = {rr*cos(ra), rr*sin(ra), 0};          // birth on the source disc
       Ray p = makeRay(p0, {0,0,1}, e);
       primary.trace(p);                               // through the optic
       if (p.getIAFlag()) slots[i] = { exit position, exit direction,
                                       p.getProb(), e };
   }

Step by step:

1. **Presumed dead.** The thread first writes a ``w = -1`` sentinel into its
   slot. If the photon is lost anywhere below, the slot keeps saying "nothing
   here" and the host skips it.
2. **The dice.** ``makeRng(seed, index)`` gives this photon its private random
   stream (reproducible on any hardware — see the primer).
3. **Energy.** With a monochromatic or uniform source every ray gets the same
   ``energy``. With a *spectrum* source, ``sampleEnergy`` throws a die into
   the cumulative-probability slots of the source spectrum: a random number
   between 0 and 1 falls into one energy's slice, and the width of each slice
   is that energy's emission probability. Strong lines get hit often, weak
   continuum rarely — exactly like the real tube.
4. **Birth.** The photon starts at a random point on the source disc
   (``sqrt`` makes the area sampling uniform), flying along the optic axis.
5. **The optic.** ``primary.trace(p)`` pushes the photon through the
   polycapillary — the heart of the beam stage, unpacked below.
6. **Verdict.** ``getIAFlag()`` true means *transmitted*: the slot records
   where the photon left the optic, in which direction, its accumulated
   transmission weight, and its energy.

Inside ``PolyCap::trace`` — reflect, absorb, or escape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A polycapillary optic is a bundle of ~100 000 hollow glass channels, each bent
so X-rays entering it are guided by grazing-incidence reflection toward the
focus. ``trace()`` walks one photon through that geometry, and there are many
ways to die:

.. code-block:: cpp

   if (dnorm < 1e-12 || ph.dz <= 0.0) { ray.setIAFlag(false); return; }  // flying backwards
   ...
   if (!withinHex(ext0, xe, ye))      { ray.setIAFlag(false); return; }  // misses the aperture
   ...
   if ((xe-cx0)*(xe-cx0) + (ye-cy0)*(ye-cy0) > rc0*rc0)
                                      { ray.setIAFlag(false); return; }  // hits the glass wall
                                                                         // between channels

A photon that makes it into an open channel then enters the **reflection
loop**: the code finds the next point where its straight path meets the
channel wall, and applies one reflection there:

.. code-block:: cpp

   double Rs, Rp;                                   // Fresnel reflectivities
   fresnelR(cos_alfa, delta_v, beta_v, Rs, Rp);     // ⊥ and ∥ polarization
   double dw = debyeWaller(roughness_, cos_alfa, ph.energy);

   ph.weight *= (Rs*frac_s + Rp*frac_p) * dw;       // pay the reflection toll
   if (ph.weight < PC_WMIN) return false;           // → absorbed

Each bounce multiplies the photon's *weight* — its survival probability — by
the Fresnel reflectivity of glass at that grazing angle and energy, times a
Debye–Waller factor for surface roughness. Steep angles and high energies
reflect poorly; that is why hard X-rays are transmitted worse, and why a
spectrum softens as it passes the optic. When the accumulated weight drops
below a threshold the photon is declared absorbed. Two more exits: a hit
*outside* the optic envelope (escaped through the side wall), and — the good
one — no further wall hit, meaning the photon flies out of the exit and its
position is projected onto the exit plane.

**"Thrown away and respawned."** Note that nothing is literally respawned: of
the 2 million photons in a chunk, those that die simply leave their ``w = -1``
sentinel, and the host keeps the survivors:

.. code-block:: cpp

   slots.toHost();
   for (long i = 0; i < n; ++i)
       if (slots.host()[i].w >= 0) beam.push_back(slots.host()[i]);

With the nist-1107 PC-236 optic roughly 1–2 % survive, so 10⁷ primaries yield
~10⁵ **exit rays** — the beam that every scan position will reuse.

.. admonition:: Where this runs

   The reflection loop is the deepest device code in voxTrace: dozens of
   double-precision bounces per photon, no memory traffic beyond the ~40
   numbers of the optic itself. It scales almost perfectly with thread count.

Stage 2 — the scan: one kernel call per (position, beam ray)
------------------------------------------------------------

Now the sample enters. For each of the scan positions from ``Simulation.txt``
the host builds two ``OpticFrame`` objects — coordinate transforms that place
the primary's focus and the secondary's focus at the current scan point
``P0`` — and launches ``ScanKernel`` over all beam rays:

.. code-block:: cpp

   for (int di = 0; di < (int)C.sc.points.size(); ++di) {
       vt::OpticFrame primFrame(P0, d_prim, zFocPrim);
       vt::OpticFrame secFrame (P0, d_sec, -aimDist);
       vt::forRange("voxTrace::scan", beamN, k);       // one thread per beam ray
       ...
   }

Moving the "measurement point" therefore never moves the sample or re-traces
the beam — it only changes the transform between optic coordinates and sample
coordinates. Inside the kernel, one thread takes one beam ray through seven
steps, each of which can end the story:

**(1) Into the sample.** The exit ray is transformed into the sample frame and
intersected with the surface plane:

.. code-block:: cpp

   primFrame.toSample(er.pos, er.dir, ps, ds);
   if (ds.z <= 0) return;                    // flying away from the surface
   double t = (z0 - ps.z) / ds.z;
   Vec3 entry = ps + ds * t;
   if (entry.x <= x0 || entry.x >= x0 + LX || ...) return;   // misses the sample

**(2) Where does it interact?** The photon penetrates the voxel grid until its
randomly drawn *optical depth* is used up — Beer–Lambert absorption, voxel by
voxel, honouring each voxel's own material:

.. code-block:: cpp

   double tau = -log(xi);                    // how much attenuation this photon survives
   while (vox >= 0) {
       double len = v.intersect(r);          // path length through this voxel
       double mu  = mats[v.getMaterialIdx()].CS_Tot_Lin(E, elems);
       if (acc + mu*len >= tau) { P = entry + dir*l; return true; }   // interacts here
       acc += mu * len;
       vox = v.getNN(r.getNextVoxel());      // step to the neighbour voxel
   }
   return false;                             // passed straight through → discarded

This is the voxel walk: ``getNN`` follows the precomputed 27-neighbour table,
so stepping across the grid costs one array lookup per voxel. A photon that
crosses the whole sample without interacting is discarded.

**(3) Which way does the new photon fly?** Here voxTrace's two Monte-Carlo
modes split (the ``vr`` switch — *variance reduction*):

.. code-block:: cpp

   if (vr) {   // importance sampling: aim at the collection window, carry a weight
       Vec3 aim = aimCtr + secFrame.u*(ar*cos(aa)) + secFrame.v*(ar*sin(aa));
       eDir  = norm(aim - P);
       wEmit = (PI_D*rWin*rWin*fabs(dot(eDir, d_sec))) / (4.0*PI_D*r2);
   } else {    // analog MC: emit isotropically, let geometry decide
       eDir = random direction on the sphere;
   }

Real fluorescence is emitted in all directions, but almost none of those
directions reach the tiny secondary optic. *Importance sampling* cheats
honestly: it always aims at a random point on the collection window and
multiplies the photon's weight by the (small) probability that isotropic
emission would have gone there. *Analog* mode emits truly isotropically and
throws the photon away if it misses — physically literal, statistically far
more expensive. Both modes discard photons emitted downward into the sample
half-space that can never exit (``eDir.z >= 0``).

**(4) What kind of interaction?** The element is chosen with probability
proportional to its share of the interaction cross-section, then the physics
type:

.. code-block:: cpp

   int type = el.getInteractionType((float)energy, rng.frand());
   if (type == 0) {                                   // photoelectric → fluorescence
       int shell = el.getExcitedShell((float)energy, rng.frand());
       if (rng.frand() >= el.Fluor_Y(shell)) return;  // Auger electron — no photon
       Ef = el.Line_Energy(el.getTransition(shell, rng.frand()));
   } else {                                           // scatter
       Ef = (type == 1) ? energy                      // Rayleigh: same energy
                        : el.getComptEnergy(energy, th);   // Compton: shifted
   }

Photoelectric absorption excites a shell; the atom relaxes either by emitting
a characteristic fluorescence photon (**this is the signal** — e.g. Cu Kα at
8.05 keV) or an Auger electron, in which case the story ends. Rayleigh scatter
re-emits at the same energy; Compton at a reduced, angle-dependent energy —
these produce the background under the peaks. If polarization is enabled, the
scatter weight additionally depends on the angle between scatter plane and the
X-ray's electric field (``wPol``).

**(5) Getting out.** The new photon must escape the sample; the same voxel
walk integrates the optical depth from the interaction point to the surface:

.. code-block:: cpp

   double tauOut = opticalDepthOut(grid, P, eDir, (float)Ef);
   if (vr) wSelf = exp(-tauOut);                    // weight mode
   else if (rng.frand() >= exp(-tauOut)) return;    // analog: survive or die

This *self-absorption* is what makes confocal depth profiling quantitative:
photons born deeper pay exponentially more to get out, and low-energy lines
pay more than high-energy ones.

**(6) The confocal filter.** The escaped photon is transformed into the
secondary optic's frame and traced through it — the same ``PolyCap::trace``
as in stage 1, in reverse mounting:

.. code-block:: cpp

   secFrame.toOptic(P, eDir, po, doo);
   Ray sray = makeRay(po, doo, Ef);
   secondary.trace(sray);
   if (!sray.getIAFlag()) return;                   // rejected by the optic
   wSec = sray.getProb();

This is the step that makes the setup *confocal*: the secondary optic only
accepts photons coming from near its focal spot. Fluorescence born outside the
overlap of the two foci is geometrically rejected here, which is why scanning
the sample through the focus produces a depth-resolved signal.

**(7) The detector.** What survives hits the Si(Li) crystal model:

.. code-block:: cpp

   float Emeas = detector.detect((float)Ef, rng, detElems, wDet);
   ...
   double w = wBeam * wEmit * wPol * wSelf * wSec * wDet;
   int b = (int)(Emeas / eBin);
   slots[bi] = Event{di, voxIdx, b, (float)w};

``detect()`` plays the detector physics: transmission through the Be window
and dead layer, absorption in the active crystal (the efficiency ``wDet``),
possibly a **Si escape peak** (the crystal's own Si Kα fluorescence carries
1.74 keV away), a Compton recoil for scattered photons, and finally Gaussian
energy broadening — the finite resolution that turns sharp lines into peaks.
The measured energy falls into a spectrum channel ``b``, and the thread writes
its one ``Event``: *scan position, voxel of origin, channel, weight*. The
weight is the product of every survival probability collected along the way —
one number that says "this photon story represents this much real intensity."

.. admonition:: Where this runs

   Steps (1)–(7) are one device kernel — one beam ray per thread, every step a
   possible early ``return``. A typical position keeps a fraction of a percent
   of the threads to the end; on a GPU the dead lanes idle within their group,
   which is why the cascade-of-exits style is the right shape for this
   hardware.

Stage 3 — from events to spectra
--------------------------------

Back on the host, the events are summed into per-position spectra
(``raw[e.pos][e.ch] += e.w``). Because the kernel weights are per-photon
probabilities, the host scales them to real counts with an exposure factor
and, for synthetic measurements, adds Poisson counting noise — the same noise
a real detector produces:

.. code-block:: cpp

   for (auto& s : raw)
       for (double& v : s)
           v = (v > 0) ? (double)std::poisson_distribution<long>(v)(noise) : 0.0;

The result is written to ``test-data/out/recon_spectra.csv`` — one energy
column plus a (measured, simulated) pair per scan position; this is the file
the voxTrace UI loads into its Spectra Explorer. With ``fit = 1`` the run
continues into the reconstruction: per-voxel weights are optimized (L-BFGS via
the ensmallen-backed OptimizerAPI) until the simulated spectra match the
measured ones — see :doc:`architecture` for that half of the story.

The complete life of one detected photon
----------------------------------------

    Born on the source disc with a dice-drawn energy → guided by ~30 grazing
    reflections through the primary polycapillary (paying Fresnel ×
    Debye–Waller at each) → enters the sample, walks the voxel grid, is
    absorbed 12 µm deep in a brass voxel → excites Cu, survives the Auger
    dice, becomes an 8.05 keV Kα photon aimed at the secondary optic → pays
    ``exp(−τ)`` self-absorption on the way out → is accepted and guided by the
    secondary optic → passes the Be window, is absorbed in the Si crystal,
    lands (resolution-broadened) in channel 161 → one ``Event`` with the
    product of all its survival weights.

Multiply by ten million primaries, and the histogram of those events *is* the
simulated spectrum.
