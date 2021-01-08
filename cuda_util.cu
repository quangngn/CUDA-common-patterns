#include <stdio.h>
#include <iostream> 
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define RAND_INIT 1
#define MALLOC_AND_RAND_INIT 2
#define RAND_10 10

#define cudaErrorCk(ans) {gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort = true) 
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr, "GPUassert: %s %s %d \n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void init_array_cpu(int *arr, int size, int type) 
{
    if (type == RAND_INIT) 
    {
        for(int i = 0; i < size; i++) {
            arr[i] = rand();
        }
    } else if (type == RAND_10)
    {
        for(int i = 0; i < size; i++) {
            arr[i] = rand() % 10;
        }
    }
}

bool compare_arrays(int * A, int * B, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (A[i] != B[i])
        {
            return false;
        }
    }
    return true;
}