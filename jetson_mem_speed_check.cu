#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_util.cu"

int main()
{
    clock_t start, end;

    // init size
    int size = 1 << 24;
    size_t byte_size = size * sizeof(int);

    int *arr;
    arr = (int *)malloc(byte_size);
    init_array_cpu(arr, size, RAND_INIT);

    int *d_arr;

    // time cudaMalloc
    start = clock();
    cudaErrorCk(cudaMalloc((int **)&d_arr, byte_size));
    end = clock();
    printf("Cuda malloc takes %ld\n", end - start);

    // time cudaMemcpy
    start = clock();
    cudaErrorCk(cudaMemcpy(d_arr, arr, byte_size, cudaMemcpyHostToDevice));
    end = clock();
    printf("Cuda memcpy host to device takes %ld\n", end - start);

    // time cudaMemcpy
    start = clock();
    cudaErrorCk(cudaMemcpy(arr, d_arr, byte_size, cudaMemcpyDeviceToHost));
    end = clock();
    printf("Cuda memcpy device to host takes %ld\n", end - start);

    // what about malloc pinned mem
    int *pinned_arr;
    start = clock();
    cudaErrorCk(cudaMallocHost((int **)&pinned_arr, byte_size));
    end = clock();
    printf("Cuda malloc pinned takes %ld\n", end - start);
}