#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_util.cu"

// In this file, we experiment with the scan algorithm.
// For example, given an array [x1, x2, x3, ... , xn].
// The inclusive scan algo is going to be [x1, (x1 + x2), (x1 + x2 + x3), ... , (x1 + x2 + ... xn)]
// The exclusive scan algo is going to be [0, x1, (x1 + x2), ... , (x1 + x2 + ... x(n-1))]

// This algorithm seem to be so good in serial manner that it is tough to be parallelized.

void scan_inclusive(int * src, int * dest, int size)
{
    int current_sum = 0
    for(int i = 0; i < size; i ++)
    {
        current_sum += src[i];
        dest[i] = current_sum;
    }
}

void scan_exclusive(int * src, int * dest, int size)
{
    int current_sum = 0
    for(int i = 0; i < size; i ++)
    {
        dest[i] = current_sum;
        current_sum += src[i];
    }
}

__global__ void gpu_excl_scan(int * src, int size)
{
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid >= size) return;

    // up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // sum up value at odd indices
        int dest_ind = (tid + 1) * stride * 2 - 1;
        if (dest_ind < blockDim.x && dest_ind - stride > 0)
        {
            src[dest_ind] = src[dest_ind - stride];
        }
        __syncthreads();
    }

    // down-sweep phase
    if (tid == 0)
        src[blockDim.x - 1] = 0;
    
    int temp = 0;
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        int index_from_end = blockDim.x - 1 - tid;

        int dest_ind = index_from_end - stride;

        if (index_from_end < blockDim.x && dest_ind > 0)
        {
            temp = src[index_from_end];
            src[index_from_end] += src[dest_ind];   // right child
            src[dest_ind] = temp;                   // left child
        }
        __syncthreads();
    }

    for (int stride = blockDim.x / 4; stride > 0; stride /= 2)
    {
        int dest_index = (tid + 1) * 2 * stride - 1;

        if (dest_ind + stride < blockDim.x)
        {
            src[dest_ind + stride] += src[dest_ind];
        }
        __syncthreads();
    }
}

// Both algorithm can be faster with shared mem
__global__ void gpu_incl_scan(int * src, int size)
{
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid >= size) return;

    // up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // sum up value at odd indices
        int dest_ind = (tid + 1) * stride * 2 - 1;
        if (dest_ind < blockDim.x && dest_ind - stride > 0)
        {
            src[dest_ind] = src[dest_ind - stride];
        }
        __syncthreads();
    }

    // down-sweep phase
    
}

int main()
{
    // init size
    int size = 1 << 12;
    int byte_size = size * sizeof(int);
    int *temp = (int *)malloc(byte_size);

    // cpu scan
    int *src = (int *)malloc(byte_size);
    int *incl_dest = (int *)malloc(byte_size);
    int *excl_dest = (int *)malloc(byte_size);

    init_array_cpu(src, size, RAND_10);

    scan_inclusive(src, incl_dest, size);
    scan_exclusive(src, excl_dest, size);

    // gpu 
    block

    free(src);
    free(incl_dest);
    free(excl_dest);
    free(temp);
}