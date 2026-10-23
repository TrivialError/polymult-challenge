// CPU emulation of the small subset of CUDA that solve.cu uses.
//
// Only for local verification on machines without an NVIDIA GPU: compiling
// solve.cu as plain C++ (clang++ -x c++) pulls this header in instead of
// cuda_runtime.h. Kernels execute block by block with a single "thread" per
// block. That is exact for solve.cu's kernels because every one of them is
// written as a sequence of FOR_THREADS loops separated by __syncthreads(), with
// no per-thread state carried across a barrier and no block reading what
// another block of the same launch writes.
//
// Streams and events are serial: every async call completes before it returns.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

struct dim3 {
    unsigned int x = 1, y = 1, z = 1;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};

inline dim3 threadIdx(0, 0, 0);
inline dim3 blockIdx(0, 0, 0);
inline dim3 blockDim(1, 1, 1);
inline dim3 gridDim(1, 1, 1);

alignas(16) inline unsigned char pm_emu_smem[1 << 20];

#define __global__
#define __device__
#define __host__
#define __shared__
#define __forceinline__ inline
#define __restrict__
#define __align__(n)

struct alignas(16) uint4 {
    uint32_t x, y, z, w;
};

inline void __syncthreads() {}
inline int __clz(int x) { return x ? __builtin_clz((unsigned)x) : 32; }
inline uint64_t __umul64hi(uint64_t a, uint64_t b) {
    return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) >> 64);
}

template <class F>
void emu_launch(dim3 grid, F&& body) {
    gridDim = grid;
    blockDim = dim3(1, 1, 1);
    threadIdx = dim3(0, 0, 0);
    for (unsigned int y = 0; y < grid.y; ++y)
        for (unsigned int x = 0; x < grid.x; ++x) {
            blockIdx = dim3(x, y, 0);
            body();
        }
}

typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
enum { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
enum { cudaDeviceScheduleSpin = 1, cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2, cudaEventDefault = 0,
       cudaHostAllocDefault = 0, cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };

inline const char* cudaGetErrorString(cudaError_t) { return "emulated cuda error"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaSetDeviceFlags(unsigned) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
template <class T>
cudaError_t cudaMalloc(T** p, size_t n) { *p = static_cast<T*>(std::malloc(n ? n : 1)); return cudaSuccess; }
template <class T>
cudaError_t cudaHostAlloc(T** p, size_t n, unsigned) { *p = static_cast<T*>(std::malloc(n ? n : 1)); return cudaSuccess; }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { *e = nullptr; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { *ms = 0; return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned = 0) { return cudaSuccess; }
template <class F>
cudaError_t cudaFuncSetAttribute(F, int, int) { return cudaSuccess; }
