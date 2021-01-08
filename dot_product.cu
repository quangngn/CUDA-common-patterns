#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_util.cu"

#define BLOCK_SIZE 1024

// assert the both v1 and v2 has same size
int dot_product_cpu (int * v1, int * v2, int size) 
{
    int result = 0;
    for (int i = 0; i < size; i++)
    {
        result += v1[i] * v2[i];
    }
    return result;
}

// kernel 1
__global__ void dot_product_gpu1(int * v1, int * v2, int * result, int size)
{
    __shared__ int smem[BLOCK_SIZE];

    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int gid = bid * threads_per_block + tid;

    if (gid >= size)
        return;

    if (tid < BLOCK_SIZE)
        smem[tid] = v1[gid] * v2[gid];
    __syncthreads();

    // reduce
    if (tid < 512)
    {
        smem[tid] += smem[tid + 512];
    }
    __syncthreads();

    if (tid < 256)
    {
        smem[tid] += smem[tid + 256];
    }
    __syncthreads();

    if (tid < 128)
    {
        smem[tid] += smem[tid + 128];
    }
    __syncthreads();

    if (tid < 64)
    {
        smem[tid] += smem[tid + 64];
    }
    __syncthreads();

    if (tid < 32)
    {
        volatile int *wmem = smem;
        wmem[tid] += wmem[tid + 32];
        wmem[tid] += wmem[tid + 16];
        wmem[tid] += wmem[tid + 8];
        wmem[tid] += wmem[tid + 4];
        wmem[tid] += wmem[tid + 2];
        wmem[tid] += wmem[tid + 1];
    }

    if (tid == 0)
    {
        atomicAdd(result, smem[0]);
    }
}

// main 
int main(int argc, char **argv)
{
    clock_t start, end;
    // init size
    int size = 1 << 24;
    size_t byte_size = size * sizeof(int);
    
    // cpu 
    start = clock();
    int *v1 = (int *)malloc(byte_size);
    int *v2 = (int *)malloc(byte_size);

    init_array_cpu(v1, size, RAND_10);
    init_array_cpu(v2, size, RAND_10);

    int dot_prod_result = dot_product_cpu(v1, v2, size);
    end = clock();
    printf("The CPU execution time is: %ld \n", end - start);

    // gpu
    // the idea is that, within one block, we can sum up the result. Then we add the final result of that block to a variable
    // No streamming is used

    // kernel 1
    start = clock();
    dim3 block(BLOCK_SIZE);
    dim3 grid(size/BLOCK_SIZE);

    int *dv1, *dv2;
    int *result;
    int temp = 0;
    cudaErrorCk(cudaMalloc((int **)&dv1, byte_size));
    cudaErrorCk(cudaMalloc((int **)&dv2, byte_size));
    cudaErrorCk(cudaMalloc((int **)&result, sizeof(int)));

    cudaErrorCk(cudaMemcpy(dv1, v1, byte_size, cudaMemcpyHostToDevice));
    cudaErrorCk(cudaMemcpy(dv2, v2, byte_size, cudaMemcpyHostToDevice));

    dot_product_gpu1<<<grid, block>>>(dv1, dv2, result, size);
    cudaDeviceSynchronize();

    end = clock();
    printf("The GPU naive execution time is: %ld \n", end - start);

    // copy result back to device
    cudaErrorCk(cudaMemcpy(&temp, result, sizeof(int), cudaMemcpyDeviceToHost));
    if (temp == dot_prod_result)
    {
        printf("Same result of dot product from the first kernel\n");
    } else
    {
        printf("Error different result of the first kernel\n");
    }

    // free device mem
    cudaErrorCk(cudaFree(dv1));
    cudaErrorCk(cudaFree(dv2));
    cudaErrorCk(cudaFree(result));
    
    // kernel 2 more like the same kernel but with stream

    start = clock();
    int n_grid = 8; // each grid is going to be launched in a different stream;
    dim3 block2(BLOCK_SIZE);
    dim3 grid2(size/(BLOCK_SIZE * 8));

    // create mem
    int *dv1_2, *dv2_2;
    int *result_2;
    int temp_2 = 0;

    cudaErrorCk(cudaMallocHost((int **)&dv1_2, byte_size));
    cudaErrorCk(cudaMallocHost((int **)&dv2_2, byte_size));
    cudaErrorCk(cudaMalloc((int **)&result_2, sizeof(int)));

    cudaStream_t odd_str;
    cudaStreamCreate(&odd_str);
    cudaStream_t even_str;
    cudaStreamCreate(&even_str);

    int offset = 0;
    int number_threads_per_grid = size / n_grid;

    for (int i = 0; i < n_grid; i++)
    {
        int *dv1_start = dv1_2 + offset;
        int *dv2_start = dv2_2 + offset;
        int *v1_start = v1 + offset;
        int *v2_start = v2 + offset;

        // Notice mind the offset when you partition the job. However, in this case we don't have to care much.
        if (i % 2 == 0)
        {
            cudaErrorCk(cudaMemcpyAsync(dv1_start, v1_start, byte_size/n_grid, cudaMemcpyHostToDevice, even_str));
            cudaErrorCk(cudaMemcpyAsync(dv2_start, v2_start, byte_size/n_grid, cudaMemcpyHostToDevice, even_str));

            dot_product_gpu1<<<grid2, block2, 0, even_str>>>(dv1_start, dv2_start, result_2, number_threads_per_grid);
        } else 
        {
            cudaErrorCk(cudaMemcpyAsync(dv1_start, v1_start, byte_size/n_grid, cudaMemcpyHostToDevice, odd_str));
            cudaErrorCk(cudaMemcpyAsync(dv2_start, v2_start, byte_size/n_grid, cudaMemcpyHostToDevice, odd_str));

            dot_product_gpu1<<<grid2, block2, 0, odd_str>>>(dv1_start, dv2_start, result_2, number_threads_per_grid);
        }
        offset += number_threads_per_grid;
    }

    // synchronize streams and destroy them after we finish
    cudaStreamSynchronize(odd_str);
    cudaStreamSynchronize(even_str);

    cudaStreamDestroy(odd_str);
    cudaStreamDestroy(even_str);

    end = clock();
    printf("The GPU naive execution time is: %ld \n", end - start);
    
    // copy result back to device
    cudaErrorCk(cudaMemcpy(&temp_2, result_2, sizeof(int), cudaMemcpyDeviceToHost));
    if (temp == dot_prod_result)
    {
        printf("Same result of dot product from the first kernel\n");
    } else
    {
        printf("Error different result of the first kernel\n");
    }

    // free device mem
    cudaErrorCk(cudaFreeHost(dv1_2));
    cudaErrorCk(cudaFreeHost(dv2_2));
    cudaErrorCk(cudaFree(result_2));

    free(v1);
    free(v2);

    return 0;
}