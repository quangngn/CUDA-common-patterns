#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_util.cu"


// Short answer, the program goes on just fine

__global__ void syncthreads_check()
{
    int tid = threadIdx.x;

    int sum = 0;
    for (int i = 0; i < 100; i++)
    {
        sum += 1;

        if (sum > 10 && tid > 10)
        {
            break;
        }
        __syncthreads(); // would this create a deadlock, some threads already exit out of the loop, while some other are still executing.
    }
}

int main()
{
    dim3 block(32);
    dim3 grid(1);
    
    printf("The kernel is started!\n");

    syncthreads_check<<<grid, block>>>();
    cudaDeviceSynchronize();

    printf("The kernel exit just fine!\n");

    return 0;
}