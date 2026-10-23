# Negacyclic polynomial multiplication — pipelined RNS/NTT on the GPU

An implementation of `polynomial-multiplication/negacyclic@1.0.0`:
`c = a·b mod (X^N + 1, q)` with `W`-bit coefficients stored as `L` little-endian
`u32` limbs. It targets the FHERMA challenge points N=65536/W=1747 (ranked) and
N=32768/W=868. `N`, `W`, `L` and `q` are runtime parameters. Any power-of-two
`N` and any `W` work, provided `q` is the spec-derived prime.

## Algorithm

1. **RNS decomposition.** Each coefficient is reduced modulo `K` 31-bit primes
   `p_j ≡ 1 (mod 2N)` with `∏ p_j > 8·N·q²`. For the ranked point, `K = 114`.
   This is a small integer GEMM over 16-bit halves of the limbs, finished with
   one Montgomery reduction.
2. **Negacyclic NTT per prime.** Cooley–Tukey forward and Gentleman–Sande
   inverse, with the `ψ` twist merged into the twiddles. Both directions share
   one Montgomery twiddle table, because `ψ^{-bitrev(2^s+i)} = −ψ^{bitrev(2^s+2^s−1−i)}`.
   The transform is split 4-step style into a *column pass* (stages
   `0..w1−1`) and a *row pass* (stages `w1..logN−1`).
3. **Exact CRT straight to mod q.** With `y_j = x_j·(M/p_j)^{-1} mod p_j` and
   `v = round(Σ y_j/p_j)` (exact in fp32, since `|x| < M/8`):
   `c ≡ Σ_j y_j·[(M/p_j) mod q] + [(−v·M) mod q]  (mod q)`. This is a second
   small GEMM. The final reduction uses `2^W ≡ δ (mod q)`, where `δ = 2^W − q`
   is tiny by the definition of `q`.

## Pipeline (why it is structured like this)

The timed call moves 2·N·L·4 bytes in and N·L·4 bytes out: 43 MB over PCIe at
the ranked point. That transfer is the floor, so the goal is to hide everything
else behind it.

- **Inputs** are staged from the caller's pageable buffers into pinned memory by
  a pool of worker threads. They go in **column chunks** (8 per operand), not
  contiguous ranges, because the first NTT pass only mixes coefficients within a
  column. Each chunk's H2D copy, RNS decomposition and forward column pass
  overlap with the transfer of the next chunk. Once `a` is complete, its row
  pass runs while `b` is still arriving.
- After the last chunk of `b`, **one fused kernel** does `b`'s forward row
  stages, the pointwise product and the inverse row stages.
- **Output** is produced in 16 column chunks. For each one: inverse column
  pass, CRT, then D2H into pinned memory, overlapped with the next chunk. The
  workers scatter each finished chunk into the result vector.
- The result `std::vector` is allocated and its pages touched by a helper
  thread outside the call. This is a buffer only; it never sees operand data.

## Files

| File | |
|---|---|
| `solve.cu` | the implementation (`fherma_init` / `fherma_run` / `fherma_free`) |
| `pm_emu.h` | CPU emulation of the CUDA subset used, for verification without a GPU |
| `tests/oracle.py` | generator + exact verifier (bigint NTT mod q, cross-checked against schoolbook) |
| `tests/run_emu.sh` | builds `solve.cu` against the emulator and checks it on many points |
| `main.cpp`, `fherma.h`, `fherma.toml`, `CMakeLists.txt` | harness and build, from `fherma-ai/polynomial-mult-cupqc-example` |

## Verification

The kernels are written as `FOR_THREADS` loops separated by `__syncthreads()`,
and no per-thread state crosses a barrier. That means the same kernel source
runs unchanged on a CPU through `pm_emu.h`, together with the real host
pipeline (thread pool, chunking, staging).

```shell
tests/run_emu.sh        # 9 points: N 2..8192, W 32..4000, random/max/sparse/small operands
tests/run_emu.sh full   # + N=32768/W=868 and N=65536/W=1747 (about 1 min)
```

All of them pass, bit-exact. Two deliberately planted bugs (too few RNS primes,
and a wrong inverse-butterfly sign) are both caught.

The emulator cannot check GPU-specific behaviour: real concurrency between
streams, shared-memory limits, or performance.

## Build and run on a GPU

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
python3 tests/oracle.py make pt 65536 1747 4 mixed   # ~1 min in Python
PM_PROFILE=1 ./build/solution pt                     # per-call phase timings on stderr
python3 tests/oracle.py verify pt
```
