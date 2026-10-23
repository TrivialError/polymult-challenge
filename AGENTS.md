# Project notes

- FHERMA challenge: negacyclic polynomial multiplication (spec `polynomial-multiplication/negacyclic@1.0.0`).
  Ranked at N=65536, W=1747 on `fherma-gpu-rtx6000` (RTX PRO 6000, Blackwell).
- Only `solve.cu` (plus headers it includes) is authored. The platform may regenerate `main.cpp`, `fherma.h`,
  and `CMakeLists.txt`, so keep all code reachable from `solve.cu` and keep the build `add_executable(solution main.cpp solve.cu)`.
- No local GPU (macOS). Verify with the CPU emulator:
  - `tests/run_emu.sh` (quick, ~15 s) and `tests/run_emu.sh full` (adds both challenge points, ~1 min).
- Kernel rule required by the emulator: kernels are sequences of `FOR_THREADS` loops separated by `__syncthreads()`.
  No per-thread state crosses a barrier, no warp intrinsics, and no block reads data written by another block of the same launch.
- `PM_PROFILE=1` prints per-call host-side phase timings to stderr.
- Git: work on feature branches; never commit to `main`.
