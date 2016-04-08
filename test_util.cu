#include "util.h"
#include <cstdio>

// Print device properties
void printDevProp(cudaDeviceProp devProp) {
    printf("Major revision number:         %lld\n",  devProp.major);
    printf("Minor revision number:         %lld\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %llu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %llu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %lld\n",  devProp.regsPerBlock);
    printf("Warp size:                     %lld\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %llu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %lld\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %lld\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %lld\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %lld\n",  devProp.clockRate);
    printf("Total constant memory:         %llu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %llu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %lld\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

void printDevices() {
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    // Iterate through devices
    for (int i = 0; i < devCount; ++i) {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
}

int main() {
    printDevices();

    testPadToMultipleOf();
    testTimer();
    testLogger();
    testSyncMemory();
    testSyncArray();

    return 0;
}
