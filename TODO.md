# voxTrace — TODO

Working list of improvements. Profiling is the current priority: the Kokkos
port needs a performance baseline on each backend before further optimisation.

## Profiling & performance

The tracer is dispatched through Kokkos (`Tracer::callTraceNewBeam`,
`PolyCap::traceBatch`), so it can be profiled with the backend-agnostic
**Kokkos Tools** as well as vendor profilers. Goal: a per-kernel baseline
(time, occupancy, memory traffic) on Serial/OpenMP and CUDA, then close the
biggest gaps.

### Build optimization (do this FIRST — any profiling before it is misleading)
- [ ] **The Makefile has no `-O` flag — everything compiles at `-O0`.** Add a
      tunable `OPT ?= -O3` to `CXXFLAGS` (and the host test recipes Test/Test2/
      Test3/Test4), keeping `make OPT=-O0` for debugging. Measured: Test-3 at 3M
      photons drops 34.6 s → 6.6 s (**5.3×**), single-threaded, no acceleration.
- [ ] After `-O3`, the remaining easy wins are multithreading (OpenMP / Kokkos
      OpenMP backend, rays are independent ≈ near-linear over the M-series cores)
      and then the GPU (Metal / Kokkos) — re-baseline profiling once `-O3` is on.

### Kokkos Tools (backend-agnostic, start here)
- [ ] Build [kokkos-tools](https://github.com/kokkos/kokkos-tools).
- [ ] **Kernel timing**: run with `KOKKOS_TOOLS_LIBS=.../libkp_kernel_timer.so`,
      then `kp_reader *.dat` — gives per-kernel call counts and time. Confirm the
      hot kernel is the per-ray `traceForward` loop.
- [ ] **Space-time stack** (`libkp_space_time_stack.so`) for a call-tree profile
      with memory high-water marks per `Kokkos::View`.
- [ ] Name every parallel region (the string arg to `parallel_for`/`parallel_reduce`)
      so the profiles are readable — audit `Tracer.cpp` and `PolyCap.hpp`.
- [ ] **Simple kernel logger** to sanity-check there are no unexpected host-device
      copies inside the trace loop.

### CUDA (NVIDIA backend)
- [ ] Build with the CUDA backend
      (`make KOKKOS_INSTALL=... KOKKOS_CXX=$(HOME)/kokkos/bin/nvcc_wrapper`).
- [ ] **Nsight Systems** (`nsys profile -t cuda,osrt ./build/src/SampleTracer ...`):
      timeline, kernel vs memcpy overlap, occupancy at a glance.
- [ ] **Nsight Compute** (`ncu --set full -k traceForward ...`): per-kernel
      registers/occupancy, warp stalls, memory throughput. Watch for register
      pressure from the large `ChemElement` tables and branch divergence in the
      interaction-type / shell-selection sampling.
- [ ] Check `Ray` struct layout (~133 B) for coalescing; consider SoA if the
      ray buffer is the bottleneck.
- [ ] Evaluate `ChemElement` table residency (`dcs_rayl[100][628]` ≈ 251 KB/elem)
      vs constant/texture memory or shared-memory staging.

### CPU (Serial / OpenMP backend)
- [ ] `perf stat` / `perf record` for IPC, cache misses, branch mispredicts.
- [ ] Scaling study over `OMP_NUM_THREADS`; verify the `parallel_reduce` in
      `PolyCap::traceBatch` scales.
- [ ] Compare `float` (current) vs the `double` PolyCap trace cost.

### Deliverable
- [ ] `docs/profiling.rst` capturing the method + baseline numbers per backend,
      and a `make profile` convenience target.

## Documentation
- [ ] Verify the Doxygen + Sphinx/Breathe build in CI (`make docs`).
- [ ] Fill remaining `@param`/`@return` tags on public methods as the API settles.
- [ ] Add a worked end-to-end example page (source → optic → sample → spectrum).

## Accelerators / portability
- GPU support is **Kokkos-only** (CPU/OpenMP/CUDA/HIP). The old Metal/MSL path
  was removed — it had drifted out of sync with the core and Metal cannot compile
  the `double` `PolyCap` trace state.
- [ ] CI matrix building Serial / OpenMP / CUDA.
- [ ] (optional) Re-introduce an Apple-GPU path — either a Kokkos backend or a
      fresh MSL kernel — but only after an all-float `PolyCap` reformulation
      (offset-from-axis, to avoid the off-axis cancellation that forces `double`;
      see `PolyCap.hpp` precision note).

## Techniques (beyond confocal µXRF)

All of these reuse the existing optical constants (δ,β from `XRayLibAPI`, as in
`PolyCap::refractiveIndex`) and the depth-resolved sample/fluorescence already in
Test-3. They split into two physics regimes: **coherent wave-optics**
(reflectivity / standing wave) and the existing **Monte-Carlo photon transport**.

- [x] **XRR** (`Test-4.cpp`): coherent Abelès transfer-matrix reflectivity of a
      layered stack. Validated: critical angles (Si 0.223°, Ni, Pt at Cu-Kα),
      total reflection below θc, Kiessig fringes (period → λ/2d at high angle).
- [ ] **GIXRF / TXRF**: the *same* transfer matrix also gives the depth/angle
      field intensity |E(z,θ)|² (the X-ray standing wave). Steps:
      1. extend the Test-4 kernel to return the down/up wave amplitudes per layer
         → `fieldIntensity(z, θ)` (factor it into a shared host header, e.g.
         `LayerStack.hpp`, since it is coherent/host-only — not a device kernel);
      2. replace Test-3's plain beam attenuation with `|E(z,θ)|²` as the
         excitation weight at each voxel depth (fluorescence + self-absorption
         out are already there);
      3. sweep the incidence angle θ → GIXRF angle curve per element; the
         total-reflection limit (substrate + trace film/particles) is TXRF;
      4. **combined GIXRF+XRR**: fit R(θ) and the element angle-curves jointly
         (reference-free depth profiling) — a `Test-5` once (1)–(3) land.

## Correctness / validation
- [ ] Wire `make test2` (polycap C library vs `PolyCap`) into CI as a regression gate.
- [ ] Golden-file test for `SampleTracer` output spectra.
