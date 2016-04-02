#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include "util.h"

#define BLOCK_SIZE 256

__global__ void vectorAdd(int numDiag, int dim, int *inMat,
        int *inDiagNums, int *inVec, int *outVec)
{
    int res = 0;
    int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
    int nPad = gridDim.x * blockDim.x;

    /* n-ty element wektora wynikowego to iloczyn skalarny n-tego wiersza
     * macierzy i wektora wejsciowego */
    for (int i = 0; i < numDiag; ++i) {
        int vecIndex = (inDiagNums[i] + threadNum) % dim;
        int matIndex = i * nPad + threadNum;
        res += inMat[matIndex] * inVec[vecIndex];
    }

    outVec[threadNum] = res;
}

int main() {
    int n, t, gridSize, nPad;
    std::cin >> n >> t;

    nPad = padToMultipleOf(n, BLOCK_SIZE);
    gridSize = nPad / BLOCK_SIZE;

    SyncArray<int> inMatrix(nPad * t);
    SyncArray<int> inDiagNums(t);
    SyncArray<int> inVector(nPad);
    SyncArray<int> outVector(nPad);

    for (int i = 0; i < t; ++i) {
        int dn;
        std::cin >> dn;
        if (dn < 0)
            inDiagNums.getHost()[i] = dn + n;
        else
            inDiagNums.getHost()[i] = dn;

        for (int j = 0; j < n; ++j)
            std::cin >> inMatrix.getHost()[i * nPad + j];
    }
    for (int i = 0; i < n; ++i)
        std::cin >> inVector.getHost()[i];

#ifdef _DEBUG
    std::cout << "Grid size: " << gridSize << std::endl;
    std::cout << "nPad: " << nPad << std::endl;
    std::cout << "DiagNums:" << std::endl;
    inDiagNums.print();
    std::cout << std::endl << "Vector:" << std::endl;
    inVector.print();
    std::cout << std::endl << "Matrix:" << std::endl;
    inMatrix.print();
    std::cout << std::endl;
#endif

    Timer tm;
    tm.startTimer();

    inMatrix.syncToDevice();
    inDiagNums.syncToDevice();
    inVector.syncToDevice();

    vectorAdd<<<gridSize, BLOCK_SIZE>>>(t, n, inMatrix.getDevice(),
         inDiagNums.getDevice(), inVector.getDevice(), outVector.getDevice());
    cudaDeviceSynchronize();
    CUDA_CALL(cudaGetLastError());

    outVector.syncToHost();

#ifdef _DEBUG
    std::cout << "time: " << tm.stopTimer() << std::endl;
#endif

    for (int i = 0; i < n; ++i)
        std::cout << outVector.getHost()[i] << std::endl;

    return 0;
}

