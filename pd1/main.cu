#include <iostream>
#include <sys/time.h>
#include "util.h"

#define BLOCK_SIZE 256

__global__ void vectorAdd(int numDiag, int *inMat,
        int *inDiagNums, int *inVec, int *outVec)
{
    int res = 0;
    int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
    int nPad = gridDim.x * blockDim.x;
    for (int i = 0; i < numDiag; ++i) {
        int diagNum = inDiagNums[i];
        res += inMat[i * nPad + threadNum] * inVec[diagNum];
    }
    outVec[threadNum] = res;
}

int main() {
    int n, t, gridSize, nPad;
    std::cin >> n >> t;
    gridSize = ((n - 1) / BLOCK_SIZE) + 1;
    nPad = gridSize * BLOCK_SIZE;

    SyncMemory<int> inMatrix(nPad * t);
    SyncMemory<int> inDiagNums(t);
    SyncMemory<int> inVector(nPad);
    SyncMemory<int> outVector(nPad);

    for (int i = 0; i < t; ++i) {
        std::cin >> inDiagNums.getHost()[i];
        for (int j = 0; j < n; ++j)
            std::cin >> inMatrix.getHost()[i * nPad + j];
    }
    for (int i = 0; i < n; ++i)
        std::cin >> inVector.getHost()[i];

    Timer tm;
    tm.startTimer();

    inMatrix.syncToDevice();
    inDiagNums.syncToDevice();
    inVector.syncToDevice();

    vectorAdd<<<gridSize, BLOCK_SIZE>>>(t, inMatrix.getDevice(),
         inDiagNums.getDevice(), inVector.getDevice(), outVector.getDevice());
    CUDA_CHECK_RETURN(cudaGetLastError());

    outVector.syncToHost();

    std::cout << "time: " << tm.stopTimer() << std::endl;

    for (int i = 0; i < n; ++i)
        std::cout << outVector.getHost()[i] << std::endl;

    return 0;
}

