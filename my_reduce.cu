#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_util.cu"

#define BLOCK_SIZE 1024

void cpu_reduce(int * arr, int size, int &sum)
{
    sum =  0;

    for (int i = 0; i < size; i++) 
    {
        sum += arr[i];
    }
}

__global__ void gpu_reduce_naive(int * arr, int size, int * sum)
{
    // this approach use interleave approach
    int bid = gridDim.x * blockIdx.y + blockIdx.x + (gridDim.x * gridDim.y * blockIdx.z);
    int tid = blockDim.x * threadIdx.y + threadIdx.x + (blockDim.x * blockDim.y * threadIdx.z);
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int gid = bid * threads_per_block + tid;

    // check if gid is out of bound
    if (gid > size) 
    {
        return;
    }

    // reduce sum
    for (int offset = threads_per_block/2; offset > 0; offset = offset/2) 
    {
        if (tid < offset) 
        {
            arr[gid] += arr[gid + offset];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        atomicAdd(sum, arr[gid]);
    }
}

template<int block_size>
__global__ void gpu_reduce_best(int * arr, int size, int * sum)
{
    // this approach use loop unrolling approach, avoid warp divergence, use dynamic parallelism,
    // use shared memory, use template variable 
    __shared__ int smem[BLOCK_SIZE];

    // this approach use interleave approach
    int bid = gridDim.x * blockIdx.y + blockIdx.x + (gridDim.x * gridDim.y * blockIdx.z);
    int tid = blockDim.x * threadIdx.y + threadIdx.x + (blockDim.x * blockDim.y * threadIdx.z);
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int gid = bid * threads_per_block + tid;

    // return if out of bound 
    if (gid >= size) 
    {
        return;
    }

    // copy to shared mem
    smem[tid] = arr[gid];
    __syncthreads();

    // this implmentation is currently work for block size = 1024
    // unrolling loops:
    if (block_size == 1024 && tid < 512)
    {
        smem[tid] += smem[tid + 512];
    }
    __syncthreads();

    if (block_size == 1024 && tid < 256)
    {
        smem[tid] += smem[tid + 256];
    }
    __syncthreads();

    if (block_size == 1024 && tid < 128)
    {
        smem[tid] += smem[tid + 128];
    }
    __syncthreads();

    if (block_size == 1024 && tid < 64)
    {
        smem[tid] += smem[tid + 64];
    }
    __syncthreads();

    if (tid < 32)
    {
        // this part is handled by a warp, so they all execute in sync by default, no thread sync needed.
        volatile int * wmem = smem;
        wmem[tid] += wmem[tid + 32];
        wmem[tid] += wmem[tid + 16];
        wmem[tid] += wmem[tid + 8];
        wmem[tid] += wmem[tid + 4];
        wmem[tid] += wmem[tid + 2];
        wmem[tid] += wmem[tid + 1];
    }

    if(tid == 0)
    {
        atomicAdd(sum, smem[0]);
    }
}

int main(int argc, char ** argv)
{
    clock_t start, end;
    // init size
    int size = 1 << 24;
    int byte_size = sizeof(int) * size;
    int temp = 0;

    // cpu reduce
    int *arr = (int *) malloc(byte_size);
    init_array_cpu(arr, size, RAND_10);

    int cpu_sum = 0;
    cpu_reduce(arr, size, cpu_sum);

    // gpu reduce
    dim3 block(BLOCK_SIZE);
    dim3 grid(size/BLOCK_SIZE);

    // kernel 1
    int *d_arr1, *sum1;
    cudaErrorCk(cudaMalloc((int **)&d_arr1, byte_size));
    cudaErrorCk(cudaMalloc((int **)&sum1, sizeof(int)));
    cudaErrorCk(cudaMemcpy(d_arr1, arr, byte_size, cudaMemcpyHostToDevice));

    // call the naive kernel
    start = clock();
    gpu_reduce_naive <<<grid, block>>>(d_arr1, size, sum1);
    cudaDeviceSynchronize();
    end = clock();

    printf("Kernel 1 time is %ld\n", end - start);

    // copy the result back to
    cudaErrorCk(cudaMemcpy(&temp, sum1, sizeof(int), cudaMemcpyDeviceToHost));

    if (temp == cpu_sum)
    {
        printf("Same result in kernel 1!\n");
    } else 
    {
        printf("Error different result in kernel 1! Expect :%d, given: %d\n", cpu_sum, temp);
    }

    // free device mem
    freeCuda(d_arr1);
    freeCuda(sum1);

    // kernel 2
    int *d_arr2, *sum2;
    cudaErrorCk(cudaMalloc((int **)&d_arr2, byte_size));
    cudaErrorCk(cudaMalloc((int **)&sum2, sizeof(int)));
    cudaErrorCk(cudaMemcpy(d_arr2, arr, byte_size, cudaMemcpyHostToDevice));

    // call the naive kernel
    start = clock();
    gpu_reduce_best <BLOCK_SIZE> <<<grid, block>>>(d_arr2, size, sum2);
    cudaDeviceSynchronize();
    end = clock();

    printf("Kernel 2 time is %ld\n", end - start);

    // copy the result back to
    cudaErrorCk(cudaMemcpy(&temp, sum2, sizeof(int), cudaMemcpyDeviceToHost));

    if (temp == cpu_sum)
    {
        printf("Same result in kernel 2!\n");
    } else 
    {
        printf("Error different result in kernel 2! Expect :%d, given: %d\n", cpu_sum, temp);
    }
    
    // free device mem
    freeCuda(d_arr2);
    freeCuda(sum2);


    // free cpu mem
    free(arr);
}