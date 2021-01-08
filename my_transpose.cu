#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_util.cu"

#define BLOCK_SIZE 1024

void cpu_transpose(int * A, int * B, int row, int col)
{
    int total_size = row * col;

    for (int i = 0; i < total_size; i++)
    {
        int c = i % row;
        int r = i / row;

        B[col * c + r] = A[i];
    }
}

__global__ void gpu_transpose_row(int * A, int * B, int row, int col)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int gid = bid * threads_per_block + tid;

    if (gid >= row * col)
    {
        return;
    }

    // conduct the transpose;
    int r = gid / row;
    int c = gid % row;

    B[c * col + r] = A[gid];
}

__global__ void gpu_transpose_col(int * A, int * B, int row, int col)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int gid = bid * threads_per_block + tid;

    if (gid >= row * col)
    {
        return;
    }

    // conduct the transpose;
    int c = gid / col;
    int r = gid % col;

    B[r * row + c] = A[gid];
}

__global__ void gpu_transpose_diagonal(int * A, int * B, int row, int col)  
{
    int diag_y = blockIdx.y;
    int diag_x = (blockIdx.x  + blockIdx.y) % gridDim.x;
    int bid = gridDim.x * gridDim.y * blockIdx.z + diag_y * gridDim.x + diag_x;

    int tid = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    int gid = bid * (blockDim.x * blockDim.y * blockDim.z) + tid;

    if (gid >= row * col) 
    {
        return;
    }

    // conduct the transpose;
    int r = gid / row;
    int c = gid % row;

    B[c * col + r] = A[gid];
}

int main()
{
    // clocks
    clock_t start, end;

    //  init size
    int row = 1 << 12;
    int col = 1 << 12;
    size_t byte_size = sizeof(int) * col * row;
    int *temp = (int *)malloc(byte_size);

    // cpu transpose
    start = clock();

    int *hA = (int *)malloc(byte_size);
    int *hB = (int *)malloc(byte_size);

    init_array_cpu(hA, row * col, RAND_INIT);

    cpu_transpose(hA, hB, row, col);
    
    end = clock();
    printf("CPU exe time: %ld\n", end - start);

    // gpu transpose
    dim3 block(BLOCK_SIZE);
    dim3 grid(row * col / BLOCK_SIZE);

    // kernel 1, row based 
    start = clock();

    int *dA1, *dB1;
    cudaErrorCk(cudaMalloc((int **)&dA1, byte_size));
    cudaErrorCk(cudaMalloc((int **)&dB1, byte_size));
    cudaErrorCk(cudaMemcpy(dA1, hA, byte_size, cudaMemcpyHostToDevice));

    gpu_transpose_row <<<grid, block>>>(dA1, dB1, row, col);

    cudaErrorCk(cudaMemcpy(temp, dB1, byte_size, cudaMemcpyDeviceToHost));
    end = clock();
    printf("GPU row major exe time: %ld\n", end - start);

    if (compare_arrays(temp, hB, row * col)) 
    {
        printf("Same result when traverse by row major!\n");
    } else 
    {
        printf("Error different result when traverse by row!\n");
    }

    // free device mem
    cudaFree(dA1);
    cudaFree(dB1);

    // kernel 2, col based
    start = clock();
    int *dA2, *dB2;
    cudaErrorCk(cudaMalloc((int **)&dA2, byte_size));
    cudaErrorCk(cudaMalloc((int **)&dB2, byte_size));
    cudaErrorCk(cudaMemcpy(dA2, hA, byte_size, cudaMemcpyHostToDevice));

    gpu_transpose_col <<<grid, block>>>(dA2, dB2, row, col);

    cudaErrorCk(cudaMemcpy(temp, dB2, byte_size, cudaMemcpyDeviceToHost));

    end = clock();
    printf("GPU col major exe time: %ld\n", end - start);

    if (compare_arrays(temp, hB, row * col)) 
    {
        printf("Same result when traverse by col major!\n");
    } else 
    {
        printf("Error different result when traverse by col!\n");
    }

    // free device mem
    cudaFree(dA2);
    cudaFree(dB2);

    // kernel 3, using diagonal coordinate based
    start = clock();
    int *dA3, *dB3;
    cudaErrorCk(cudaMalloc((int **)&dA3, byte_size));
    cudaErrorCk(cudaMalloc((int **)&dB3, byte_size));
    cudaErrorCk(cudaMemcpy(dA3, hA, byte_size, cudaMemcpyHostToDevice));

    gpu_transpose_diagonal<<<grid, block>>>(dA3, dB3, row, col);

    cudaErrorCk(cudaMemcpy(temp, dB3, byte_size, cudaMemcpyDeviceToHost));

    end = clock();
    printf("GPU diagonal coordinate exe time: %ld\n", end - start);

    if (compare_arrays(temp, hB, row * col)) 
    {
        printf("Same result when traverse by diagnonal coordinate!\n");
    } else 
    {
        printf("Error different result when traverse by diagonal coordinate!\n");
    }

    // free device mem
    cudaFree(dA3);
    cudaFree(dB3);

    free(hA);
    free(hB);
    free(temp);
    return 0;
}