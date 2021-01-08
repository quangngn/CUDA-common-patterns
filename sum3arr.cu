#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_util.cu"

__global__ void sum3Array_traverse1(int * A, int * B, int * C, int * D, int size) 
{
    // imagine the grid is a 3D cube, we first find the thread coordinate;
    int x_coord = blockIdx.x * blockDim.x + threadIdx.x;
    int y_coord = blockIdx.y * blockDim.y + threadIdx.y;
    int z_coord = blockIdx.z * blockDim.z + threadIdx.z;

    // x-y plane first then row major
    int gid = z_coord * (blockDim.x * blockDim.y * gridDim.x * gridDim.y) +
                y_coord * blockDim.x * gridDim.x + 
                x_coord;

    if (gid < size) 
    {
        D[gid] = A[gid] + B[gid] +C[gid];
    }
} 

__global__ void sum3Array_traverse2(int * A, int * B, int * C, int * D, int size)
{
    // consecutive elements are put into same block
    int bid = gridDim.x * blockIdx.y + blockIdx.x + (gridDim.x * gridDim.y * blockIdx.z);
    int tid = blockDim.x * threadIdx.y + threadIdx. x + (blockDim.x * blockDim.y * threadIdx.z);
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int gid = bid * threads_per_block + tid;

    if (gid < size) 
    {
        D[gid] = A[gid] + B[gid] +C[gid];
    }
}

void sum3Array_cpu(int * A, int  * B, int * C, int * D, int size)
{
    for (int i = 0; i< size; i++) 
    {
        D[i] = A[i] + B[i] + C[i];
    }
}

int main(int argc, char ** argv)
{   
    clock_t start, end;
    // init arr
    int size = 1 << 24;
    int byte_size = sizeof(int) * size;
    int *temp = (int *) malloc(byte_size);

    // for CPU
    start = clock();
    int *h_A = (int *) malloc(byte_size); 
    int *h_B = (int *) malloc(byte_size); 
    int *h_C = (int *) malloc(byte_size);
    int *h_D = (int *) malloc(byte_size);

    init_array_cpu(h_A, size, RAND_INIT);
    init_array_cpu(h_B, size, RAND_INIT);
    init_array_cpu(h_C, size, RAND_INIT);

    sum3Array_cpu(h_A, h_B, h_C, h_D, size);
    end = clock();

    printf("CPU time = %ld ms\n", end - start);
    
    // for GPU
    dim3 block(32,32);
    dim3 grid(8,8, 256);

    // for kernel 1
    start = clock();
    int * d_A1, *d_B1, *d_C1, *d_D1;
    cudaErrorCk(cudaMalloc((int **)&d_A1, byte_size));
    cudaErrorCk(cudaMalloc((int **)&d_B1, byte_size));
    cudaErrorCk(cudaMalloc((int **)&d_C1, byte_size));
    cudaErrorCk(cudaMalloc((int **)&d_D1, byte_size));

    cudaErrorCk(cudaMemcpy(d_A1, h_A, byte_size, cudaMemcpyHostToDevice));
    cudaErrorCk(cudaMemcpy(d_B1, h_B, byte_size, cudaMemcpyHostToDevice));
    cudaErrorCk(cudaMemcpy(d_C1, h_C, byte_size, cudaMemcpyHostToDevice));

    // call first kernel
    sum3Array_traverse1 <<<grid, block>>>(d_A1, d_B1, d_C1, d_D1, size);
    cudaDeviceSynchronize();

    // copy result back
    cudaMemcpy(temp, d_D1, byte_size, cudaMemcpyDeviceToHost);
    
    end = clock();
    printf("GPU time 1 = %ld ms\n", end - start);

    // verify result 
    if(compare_arrays(temp, h_D, size)) 
    {
        printf("Same result!\n");
    } else 
    {
        printf("Error! Different result!\n");
    }

    // free gpu mem
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    cudaFree(d_D1);

    // for kernel 2
    start = clock();

    int * d_A2, *d_B2, *d_C2, *d_D2;
    cudaErrorCk(cudaMalloc((int **)&d_A2, byte_size));
    cudaErrorCk(cudaMalloc((int **)&d_B2, byte_size));
    cudaErrorCk(cudaMalloc((int **)&d_C2, byte_size));
    cudaErrorCk(cudaMalloc((int **)&d_D2, byte_size));

    cudaErrorCk(cudaMemcpy(d_A2, h_A, byte_size, cudaMemcpyHostToDevice));
    cudaErrorCk(cudaMemcpy(d_B2, h_B, byte_size, cudaMemcpyHostToDevice));
    cudaErrorCk(cudaMemcpy(d_C2, h_C, byte_size, cudaMemcpyHostToDevice));

    // call second kernel
    sum3Array_traverse1 <<<grid, block>>>(d_A2, d_B2, d_C2, d_D2, size);
    cudaDeviceSynchronize();

    // copy result back
    cudaMemcpy(temp, d_D2, byte_size, cudaMemcpyDeviceToHost);

    end = clock();
    printf("GPU 2 time = %ld ms\n", end - start);

    // verify result 
    if(compare_arrays(temp, h_D, size)) 
    {
        printf("Same result!\n");
    } else 
    {
        printf("Error! Different result!\n");
    }

    // free gpu mem
    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);
    cudaFree(d_D2);

    // free cpu mem
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    return 0;
}