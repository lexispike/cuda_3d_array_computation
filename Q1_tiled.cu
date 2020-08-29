/*
 * Alyxandra Spikerman
 * High Perfomance Computing
 * Homework 6 - Question 1
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define TILE_SIZE 2
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
    int i = blockIdx.x * TILE_SIZE + threadIdx.x;
    int j = blockIdx.y * TILE_SIZE + threadIdx.y;
    int k = blockIdx.z * TILE_SIZE + threadIdx.z;

    if (i < (n-1) && i > 0 && j < (n-1) && j > 0 && k < (n-1) && k > 0) {
        a[n*n*i + n*j + k] = 0.8 * (b[n*n*(i-1) + n*j + k] +
                                    b[n*n*(i+1) + n*j + k] +
                                    b[n*n*i + n*(j-1) + k] +
                                    b[n*n*i + n*(j+1) + k] +
                                    b[n*n*i + n*j + (k-1)] +
                                    b[n*n*i + n*j + (k+1)]);
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

    int gridSize = 1 + ((n - 1) / TILE_SIZE);
    dim3 dimGrid(gridSize, gridSize, gridSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);

    cudaMemcpy(d_a, h_a, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, total_bytes, cudaMemcpyHostToDevice);

    printf("\nStart kernel\n");

    float Ktime;
    TIMER_CREATE(Ktime);
    TIMER_START(Ktime);

    get_a<<< dimGrid, dimBlock >>>(d_a, d_b); // Execute the kernel
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
