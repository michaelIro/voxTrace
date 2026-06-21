# voxTrace — TODO

Working list of improvements. Profiling is the current priority: the Kokkos
port needs a performance baseline on each backend before further optimisation.

## Profiling & performance

The tracer is dispatched through Kokkos (`Tracer::callTraceNewBeam`,
`PolyCap::traceBatch`), so it can be profiled with the backend-agnostic
**Kokkos Tools** as well as vendor profilers. Goal: a per-kernel baseline
(time, occupancy, memory traffic) on Serial/OpenMP and CUDA, then close the
biggest gaps.

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
- [ ] Metal path for `PolyCap` needs an all-float reformulation (offset-from-axis)
      to avoid the off-axis cancellation that currently forces `double`
      (see `PolyCap.hpp` precision note).
- [ ] CI matrix building Serial / OpenMP / CUDA (and Metal on macOS).

## Correctness / validation
- [ ] Wire `make test2` (polycap C library vs `PolyCap`) into CI as a regression gate.
- [ ] Golden-file test for `SampleTracer` output spectra.
