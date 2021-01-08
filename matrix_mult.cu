#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_util.cu"

#define BLOCK_SIZE 1024

// experiment with 16 and 32
#define TILE_WIDTH 32
#define TILE_HEIGTH 32

// cpu implementation
void cpu_matrix_mult(int * A, int * B, int * C, int rowA, int colA, int rowB, int colB)
{
    for (int rA = 0; rA < rowA; rA++)
    {
        for (int cA = 0; cA < colA; cA++)
        {
            for(int cB = 0; cB < colB; cB++)
            {
                C[rA * colA + cB] += A[rA * colA + cA] * B[cA * colB + cB];
            }
        }
    }
}

__global__ void gpu_matrix_mult_naive(int * A, int * B, int * C, int rowA, int colA, int rowB, int colB)
{
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int gid = bid * BLOCK_SIZE + tid;

    if (gid >= rowA * colB)
        return;

    int r = gid / colA;
    int c = gid % colA;

    int dot_prod = 0;
    for (int i = 0; i < colA; i++)
    {
        dot_prod += A[r * colA + i] * B[i * colB + c];
    }

    C[gid] = dot_prod;
}

__global__ void gpu_matrix_mult_tile(int * A, int * B, int * C, int rowA, int colA, int rowB, int colB)
{
    // we might have bank conflict in tile B because we will read it in col
    // in this case we assume that the tile from A and B has the same dimension as the tile in C
    // in other situation, this might not be the case.
    __shared__ int tileA[TILE_HEIGTH * TILE_WIDTH];     // matrix of size TILE_HEIGTH x TILE_WIDTH
    __shared__ int tileB[TILE_WIDTH * TILE_WIDTH];      // matrix of size TILE_WIDTH x TILE_WIDTH

    int tileA_size = TILE_HEIGTH * TILE_WIDTH;
    int tileB_size = TILE_WIDTH * TILE_WIDTH;

    // int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    // this is the coordinate in the C matrix
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    int gid = ty * gridDim.x * blockDim.x + tx;

    if (gid >= rowA * colB)
        return;

    int A_row_offset = 0;
    int B_col_offset = 0;
    int tiles_per_A_row = colA / TILE_WIDTH;

    int dot_prod = 0;
    for (int tile_i = 0; tile_i < tiles_per_A_row; tile_i++)
    {
        // thread load data to shared mem
        if (tid < tileA_size)
        {
            int xA = tid % TILE_WIDTH + A_row_offset;
            int yA = tid / TILE_WIDTH + blockDim.y * blockIdx.y;
            tileA[tid] = A[yA * colA + xA];
        }

        if (tid < tileB_size)
        {
            int xB = tid % TILE_WIDTH + blockDim.x * blockIdx.x;
            int yB = tid / TILE_WIDTH + B_col_offset;
            tileB[tid] = B[yB * colB + xB];
        }
        __syncthreads();
        
        // actually do the calculation, each thread takes care of a point in C.
        for (int i = 0; i < TILE_WIDTH; i++)
        {
            dot_prod += tileA[threadIdx.y * TILE_WIDTH + i] * tileB[i * TILE_WIDTH + threadIdx.x]; // this might not be efficient
        }

        A_row_offset += TILE_WIDTH;
        B_col_offset += TILE_WIDTH;
        __syncthreads();
    }

    C[gid] = dot_prod;
}

__global__ void gpu_matrix_mult_tile_with_padding(int * A, int * B, int * C, int rowA, int colA, int rowB, int colB)
{
    // we might have bank conflict in tile B because we will read it in col
    // in this case we assume that the tile from A and B has the same dimension as the tile in C
    // in other situation, this might not be the case.
    __shared__ int tileA[TILE_HEIGTH * TILE_WIDTH]; // matrix of size TILE_HEIGTH x TILE_WIDTH
    __shared__ int tileB[TILE_WIDTH * (TILE_WIDTH + 1)]; // matrix of size TILE_WIDTH x TILE_WIDTH

    int tileA_size = TILE_HEIGTH * TILE_WIDTH;
    int tileB_true_size = TILE_WIDTH * TILE_WIDTH;

    // int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    // this is the coordinate in the C matrix
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    int gid = ty * gridDim.x * blockDim.x + tx;

    if (gid >= rowA * colB)
        return;

    int A_row_offset = 0;
    int B_col_offset = 0;
    int tiles_per_A_row = colA / TILE_WIDTH;
    // int tiles_per_B_row = rowB / TILE_WIDTH; // these two should be the same

    int dot_prod = 0;
    for (int tile_i = 0; tile_i < tiles_per_A_row; tile_i++)
    {
        // thread load data to shared mem
        if (tid < tileA_size)
        {
            int xA = tid % TILE_WIDTH + A_row_offset;
            int yA = tid / TILE_WIDTH + blockDim.y * blockIdx.y;
            tileA[tid] = A[yA * colA + xA];
        }

        if (tid < tileB_true_size)
        {
            int xB = tid % TILE_WIDTH + blockDim.x * blockIdx.x;
            int yB = tid / TILE_WIDTH + B_col_offset;
            tileB[tid + tid/TILE_WIDTH] = B[yB * colB + xB];
        }
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
        {
            dot_prod += tileA[threadIdx.y * TILE_WIDTH + i] * tileB[i * (TILE_WIDTH + 1) + threadIdx.x]; // this might not be efficient
        }

        A_row_offset += TILE_WIDTH;
        B_col_offset += TILE_WIDTH;
    }

    C[gid] = dot_prod;
}

int main(int argc, char ** argv)
{
    // init clock
    clock_t start, end;
    // init size
    int rowA = 1 << 10;
    int colA = 1 << 9;
    int rowB = 1 << 9;
    int colB = 1 << 11;

    int A_byte_size = rowA * colA * sizeof(int);
    int B_byte_size = rowB * colB * sizeof(int);
    int C_byte_size = rowA * colB * sizeof(int);
    int *temp = (int *)malloc(C_byte_size);

    // cpu
    start = clock();
    int *A = (int *)malloc(A_byte_size);
    int *B = (int *)malloc(B_byte_size);
    int *C = (int *)calloc(rowA * colB, sizeof(int));

    init_array_cpu(A, rowA * colA, RAND_10);
    init_array_cpu(B, rowB * colB, RAND_10);

    cpu_matrix_mult(A, B, C, rowA, colA, rowB, colB);
    end = clock();
    printf("CPU execution time is %ld\n\n", end - start);


    // gpu naive
    start = clock();
    dim3 block(BLOCK_SIZE);
    dim3 grid(rowA * colB / BLOCK_SIZE);

    int *dA, *dB, *dC;
    cudaErrorCk(cudaMalloc((int **)&dA, A_byte_size));
    cudaErrorCk(cudaMalloc((int **)&dB, B_byte_size));
    cudaErrorCk(cudaMalloc((int **)&dC, C_byte_size));
    cudaErrorCk(cudaMemset(dC, 0, C_byte_size));

    cudaErrorCk(cudaMemcpy(dA, A, A_byte_size, cudaMemcpyHostToDevice));
    cudaErrorCk(cudaMemcpy(dB, B, B_byte_size, cudaMemcpyHostToDevice));

    gpu_matrix_mult_naive<<<grid, block>>>(dA, dB, dC, rowA, colA, rowB, colB);
    cudaDeviceSynchronize();
    cudaErrorCk(cudaMemcpy(temp, dC, C_byte_size, cudaMemcpyDeviceToHost));
    end = clock();

    printf("GPU naive time is %ld\n", end - start);

    if (compare_arrays(temp, C, rowA * colB))
    {
        printf("Same result for gpu naive matrix mult!\n\n");
    } else {
        printf("Error different result for gpu naive matrix mult!\n\n");
    }

    cudaErrorCk(cudaFree(dA));
    cudaErrorCk(cudaFree(dB));
    cudaErrorCk(cudaFree(dC));



    // gpu tile_based
    start = clock();
    dim3 block2(TILE_WIDTH, TILE_HEIGTH); // size of a tile
    dim3 grid2(colB/TILE_WIDTH, rowA/TILE_HEIGTH);

    int *dA2, *dB2, *dC2;
    cudaErrorCk(cudaMalloc((int **)&dA2, A_byte_size));
    cudaErrorCk(cudaMalloc((int **)&dB2, B_byte_size));
    cudaErrorCk(cudaMalloc((int **)&dC2, C_byte_size));

    cudaErrorCk(cudaMemcpy(dA2, A, A_byte_size, cudaMemcpyHostToDevice));
    cudaErrorCk(cudaMemcpy(dB2, B, B_byte_size, cudaMemcpyHostToDevice));

    gpu_matrix_mult_tile<<<grid2, block2>>>(dA2, dB2, dC2, rowA, colA, rowB, colB);
    cudaDeviceSynchronize();

    cudaErrorCk(cudaMemcpy(temp, dC2, C_byte_size, cudaMemcpyDeviceToHost));
    end = clock();
    printf("CPU tile-based time is %ld\n", end - start);

    if (compare_arrays(temp, C, rowA * colB))
    {
        printf("Same result for gpu tile-based using shared memory matrix mult!\n\n");
    } else {
        printf("Error different result for gpu tile based using shared memory matrix mult!\n\n");
    }

    cudaErrorCk(cudaFree(dA2));
    cudaErrorCk(cudaFree(dB2));
    cudaErrorCk(cudaFree(dC2));



    // gpu tile_based with padding
    start = clock();
    dim3 block3(TILE_WIDTH, TILE_HEIGTH); // size of a tile
    dim3 grid3(colB/TILE_WIDTH, rowA/TILE_HEIGTH);

    int *dA3, *dB3, *dC3;
    cudaErrorCk(cudaMalloc((int **)&dA3, A_byte_size));
    cudaErrorCk(cudaMalloc((int **)&dB3, B_byte_size));
    cudaErrorCk(cudaMalloc((int **)&dC3, C_byte_size));

    cudaErrorCk(cudaMemcpy(dA3, A, A_byte_size, cudaMemcpyHostToDevice));
    cudaErrorCk(cudaMemcpy(dB3, B, B_byte_size, cudaMemcpyHostToDevice));

    gpu_matrix_mult_tile_with_padding<<<grid3, block3>>>(dA3, dB3, dC3, rowA, colA, rowB, colB);
    cudaDeviceSynchronize();
    cudaErrorCk(cudaMemcpy(temp, dC3, C_byte_size, cudaMemcpyDeviceToHost));
    end = clock();
    printf("CPU tile-based with padding time is %ld\n", end - start);

    if (compare_arrays(temp, C, rowA * colB))
    {
        printf("Same result for gpu tile-based using shared memory with padding matrix mult!\n\n");
    } else {
        printf("Error different result for gpu tile based using shared memory with padding matrix mult!\n\n");
    }
    cudaErrorCk(cudaFree(dA3));
    cudaErrorCk(cudaFree(dB3));
    cudaErrorCk(cudaFree(dC3));

    // free cpu A, B, C
    free(A);
    free(B);
    free(C);
    free(temp);
}