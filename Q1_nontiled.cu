/*
 * Alyxandra Spikerman
 * High Perfomance Computing
 * Homework 6 - Question 1
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define n 64

// taken from transpose.cu from HW5
#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);

#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \

#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);

// CUDA kernel
__global__ void get_a(float* a, float* b) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // taken from https://devtalk.nvidia.com/default/topic/619509/cuda-programming-and-performance/3d-arrays-where-to-start-/post/3968516/#3968516
    // get x,y,z with the index
    int z = id/(n*n);
    id -= z*n*n;
    int y = id/n;
    id -= y*n;
    int x = id;

    for (int i = x; i < n-1; i += stride) {
        for (int j = y; j < n-1; j += stride) {
            for (int k = z; k < n-1; k += stride) {
                if (i > 0 && j > 0 && k > 0) {
                    a[n*n*i + n*j + k] = 0.8 * (b[n*n*(i-1) + n*j + k] +
                                                b[n*n*(i+1) + n*j + k] +
                                                b[n*n*i + n*(j-1) + k] +
                                                b[n*n*i + n*(j+1) + k] +
                                                b[n*n*i + n*j + (k-1)] +
                                                b[n*n*i + n*j + (k+1)]);
                }
            }
        }
    }
}

int main(int argc, char* argv[] ) {
    size_t total_bytes = n * n * n * sizeof(float);
    float* h_a = (float*)malloc(total_bytes);
    float* h_b = (float*)malloc(total_bytes);
    float* d_a;
    cudaMalloc(&d_a, total_bytes);
    float* d_b;
    cudaMalloc(&d_b, total_bytes);

    srand(150);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                h_b[n*n*i + n*j + k] = (rand() % 50) + 1;
            }
        }
    }

    cudaMemcpy(d_a, h_a, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, total_bytes, cudaMemcpyHostToDevice);

    printf("\nStart kernel\n\n");

    float Ktime;
    TIMER_CREATE(Ktime);
    TIMER_START(Ktime);

    get_a<<< (n*n*n)/128, 128 >>>(d_a, d_b); // Execute the kernel
    cudaDeviceSynchronize(); // wait for everything to finish before accessing

    TIMER_END(Ktime);
    printf("Kernel Execution Time: %f ms\n", Ktime);

    cudaMemcpy(h_a, d_a, total_bytes, cudaMemcpyDeviceToHost); // Copy histogram to host

    // for (int i = 1; i < n - 1; i++) {
    //     for (int j = 1; j < n - 1; j++) {
    //         for (int k = 1; k < n - 1; k++) {
    //             printf("%f ", h_a[n*n*i + n*j + k]);
    //         }
    //     }
    // }

    // free allocated memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    return 0;
}
