#pragma once
#include "pch.h"

namespace myblascuda {
    template<class T>
    __global__ void dot(int n, const T* a, const T* b, T* out) {
        __shared__ T cache[NTHREADSPERBLOCK];
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        cache[threadIdx.x] = 0.0;

        while (i < n) {
            cache[threadIdx.x] += a[i] * b[i];
            i += gridDim.x * blockDim.x;
        }
        __syncthreads();  // required because later on the current thread is
                          // accessing data written by another thread
        i = NTHREADSPERBLOCK / 2;
        while (i > 0) {
            if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
            __syncthreads();
            i /= 2;  // not sure bitwise operations are actually faster
        }
        if (threadIdx.x == 0) atomicAdd(out, cache[0]);
    }

    template<class T>
    __global__ void multi(int n, float s, const T* b, T* c) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        c[i] = s * b[i];
    }

    template<class T>
    __global__ void multi(int n, double s, const T* b, T* c) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        c[i] = s * b[i];
    }


    template<class T>
    __global__ void plus(int n, const T* a, const T* b, T* c) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        c[i] = a[i] + b[i];

    }

    template<class T>
    __global__ void minus(int n, const T* a, const T* b, T* c) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        c[i] = a[i] - b[i];
    }

    template<class T>
    __global__ void to(int n, const T* a, T* b) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        b[i] = a[i];
    }

}
