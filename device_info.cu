#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main() 
{
    // get the number of devices
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    printf("The number of devices is: %d\n", devCount);

    // get info of each device
    for (int i = 0; i < devCount; i++) 
    {
        cudaDeviceProp devProb;
        cudaGetDeviceProperties(&devProb, i);

        printf("Device No. %d ******************************************\n", i);

        // print name
        printf("\tName: %s\n", devProb.name);

        // print number of multiprocessors
        printf("\tNumber of multiprocessors:            %d \n", devProb.multiProcessorCount);

        // print clock rate
        printf("\tClock rate:                           %d\n", devProb.clockRate);

        // print compute capability
        printf("\tCompute capability:                   %d.%d\n", devProb.major, devProb.minor);

        // print amount of global memory
        printf("\tTotal amount of global memory:        %.2f KB\n", devProb.totalGlobalMem / 1024.0);

        // print amount of constant memory
        printf("\tTotal amount of constant memory:      %.2f KB\n", devProb.totalConstMem / 1024.0);

        // print share memory per block 
        printf("\tAmount of shared memory per block:    %.2f KB\n", devProb.sharedMemPerBlock / 1024.0);

        // print warp size
        printf("\tWarp size:                            %d\n\n", devProb.warpSize);

        // print max dimmension
        // Thread related
        printf("\tMax threads dim:                      (%d, %d, %d)\n", devProb.maxThreadsDim[0], devProb.maxThreadsDim[1], devProb.maxThreadsDim[2]);
        printf("\tMax threads per block:                %d\n", devProb.maxThreadsPerBlock);
        printf("\tMax threads per MP:                   %d\n\n", devProb.maxThreadsPerMultiProcessor);

        // Block related
        //printf("\tMax blocks per MP:                    %d\n\n", devProb.maxBlocksPerMultiProcessor);

        // Grid related
        printf("\tMax grid size:                        (%d, %d, %d)\n", devProb.maxGridSize[0],  devProb.maxGridSize[1],  devProb.maxGridSize[2]);
    }

    return 0;
}