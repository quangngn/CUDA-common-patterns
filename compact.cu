#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_util.cu"

__global__ void pre_compact(int * src, int * temp, int size)
{
    gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid >= size) return;

    if (src[gid] != 0)
    {
        temp[gid] == 1;
    } else
    {
        temp[gid] == 0;
    }
}

__global__ void exclusive_scan_for_index(int *temp, int size)
{
    // exclusive scan here, this might need to be done in some phases
}

__global__ void compact(int * src, int * out, int * temp, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid < size)
    {
        if (src[gid] != 0)
        {
            output[temp[gid]] = src[gid];
        }
    }
}