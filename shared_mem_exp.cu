#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_util.cu"

#define BLOCK_SIZE 1024

__global__ void colRead(int * arr, int size)
{
    __shared__ int smem[BLOCK_SIZE];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int gid = tid;

    if (tid < 32 && gid < size)
    {
        smem[gid] = arr[gid];
        __syncthreads();

        // copy value back to arr in col manner;
        arr[gid] = smem[threadIdx.y + threadIdx.x * blockDim.y];
    }
}

__global__ void colReadPadded(int * arr, int size)
{
    __shared__ int smem[BLOCK_SIZE + 32];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int gid = tid;

    int smem_id = threadIdx.x * threadIdx.y * (blockDim.x + 1);

    if (tid < 32 && gid < size)
    {
        smem[smem_id] = arr[gid];
        __syncthreads();

        // copy value back to arr in col manner;
        arr[gid] = smem[threadIdx.y + threadIdx.x * (blockDim.x + 1)];
    }
}


__global__ void rowRead(int *arr, int size)
{
    __shared__ int smem[BLOCK_SIZE];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int gid = tid;

    if (tid < 32 && gid < size)
    {
        smem[gid] = arr[gid];
        __syncthreads();

        // copy value back to arr in col manner;
        arr[gid] = smem[gid];
    }
}

__global__ void colWrite(int * arr, int size) {
    __shared__ int smem[BLOCK_SIZE];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int gid = tid;

    int smem_idx = tid 
} 

int main()
{
    clock_t start, end;

    // init size
    int size = BLOCK_SIZE;
    size_t byte_size = size * sizeof(int);

    // cpu arr
    int *arr = (int *)malloc(byte_size);
    init_array_cpu(arr, size, RAND_INIT);

    // gpu 
    dim3 block(32,32);
    dim3 grid(1);

    // Col read only
    int *d_arr1;
    cudaErrorCk(cudaMalloc((int **)&d_arr1, byte_size));
    cudaErrorCk(cudaMemcpy(d_arr1, arr, byte_size, cudaMemcpyHostToDevice));

    start = clock();
    colRead<<<grid, block>>>(d_arr1, size);
    cudaDeviceSynchronize();
    end = clock();
    
    printf("Col read takes: %ld\n", end - start);

    // Row read only
    int *d_arr2;
    cudaErrorCk(cudaMalloc((int **)&d_arr2, byte_size));
    cudaErrorCk(cudaMemcpy(d_arr2, arr, byte_size, cudaMemcpyHostToDevice));

    start = clock();
    rowRead<<<grid, block>>>(d_arr2, size);
    cudaDeviceSynchronize();
    end = clock();
    
    printf("Row read takes: %ld\n", end - start);


    // Col read with padded shared mem
    int *d_arr3;
    cudaErrorCk(cudaMalloc((int **)&d_arr3, byte_size));
    cudaErrorCk(cudaMemcpy(d_arr3, arr, byte_size, cudaMemcpyHostToDevice));

    start = clock();
    colReadPadded<<<grid, block>>>(d_arr3, size);
    cudaDeviceSynchronize();
    end = clock();
    printf("Col padded read takes: %ld\n", end - start);

    int * temp1 = (int *) malloc(byte_size);
    int * temp2 = (int *) malloc(byte_size);

    cudaMemcpy(temp1, d_arr1, byte_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(temp2, d_arr3, byte_size, cudaMemcpyDeviceToHost);
    
    if(compare_arrays(temp1, temp2, size))
    {
        printf("Same result!");
    }

    return 0;
}