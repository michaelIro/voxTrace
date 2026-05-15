# Pre-build Setup

## M4 (Apple Silicon — macOS)

Install dependencies via Homebrew:

```bash
brew install pkg-config armadillo gsl ensmallen
brew tap tschoonj/tap && brew install tschoonj/tap/xraylib
```

Build host-only test targets (no Kokkos required):

```bash
make test          # builds build/src/Test
make polycap-test  # builds build/src/PolyCapTraceTest
```

For the full simulation binary with Kokkos (OpenMP backend):

```bash
make KOKKOS_PATH=~/kokkos KOKKOS_DEVICES=OpenMP
```
