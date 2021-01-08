#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_util.cu"


__global__ void grid_traverse1(int * arr, int size) {
    // imagine the grid is a 3D cube, we first find the thread coordinate;
    int x_coord = blockIdx.x * blockDim.x + threadIdx.x;
    int y_coord = blockIdx.y * blockDim.y + threadIdx.y;
    int z_coord = blockIdx.z * blockDim.z + threadIdx.z;

    // x-y plane first then row major
    int gid = z_coord * (blockDim.x * blockDim.y * gridDim.x * gridDim.y) +
                y_coord * blockDim.x * gridDim.x + 
                x_coord;

    if (gid < (1<<24)) {
        if (gid == 1<<24 - 1) {
            printf("reach final!\n");
        }
        int num = arr[gid];

        int z = num / (blockDim.x * blockDim.y * gridDim.x * gridDim.y);
        int remain = num % (blockDim.x * blockDim.y * gridDim.x * gridDim.y);
        int y = remain / (blockDim.x * gridDim.x);
        int x = remain % (blockDim.x * gridDim.x);

        if (x != x_coord || y != y_coord || z != z_coord) {
            printf("encode error \n");
        }
    }
}

__global__ void grid_traverse2(int * arr, int size) {
    int bid = gridDim.x * blockIdx.y + blockIdx.x + (gridDim.x * gridDim.y * blockIdx.z);
    int tid = blockDim.x * threadIdx.y + threadIdx. x + (blockDim.x * blockDim.y * threadIdx.z);

    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;

    int gid = bid * threads_per_block + tid;

    if (gid < (1<<24)) {
        if (gid == 1<<24 - 1) {
            printf("reach final!\n");
        }
        int num = arr[gid];

        // random check
        if (bid == 10 && tid < 15) {
            printf("thread id = %d, with num = %d\n", tid, num);
        }
    }
}

int main(int argc, char ** argv) {
    int traverse_style = 1;

    if (argc > 1) {
        traverse_style = atoi(argv[1]);
    }
 
    // init array 
    int size = 1 << 24;
    int *arr = (int*) malloc(sizeof(int) * size);
    for (int i = 0; i < size; i++) {
        arr[i] = i;
    }

    // move to device
    int *d_arr;
    cudaErrorCk(cudaMalloc((int **)&d_arr, sizeof(int) * size));
    cudaErrorCk(cudaMemcpy(d_arr, arr, sizeof(int) * size, cudaMemcpyHostToDevice));

    // First way to distribute the work, we imagine the it is a cube of 256*256*256 threads.
    // grid = 4x4x4 (blocks)
    // block = 64x64x64
    
    dim3 block(32, 32);
    dim3 grid(8, 8, 256);

    if (traverse_style == 1) {
        // solution 1: the array is distribute across the dimension
        grid_traverse1 <<<grid, block>>>(d_arr, size);
        cudaDeviceSynchronize();
    } else if (traverse_style == 2) {
        // solution 2: consecutive element in an array are staying in the same block
        grid_traverse2 <<<grid, block>>>(d_arr, size);
        cudaDeviceSynchronize();
    }

    // solution 2: 

    free(arr);
    cudaErrorCk(cudaFree(d_arr));

    return 0;
}