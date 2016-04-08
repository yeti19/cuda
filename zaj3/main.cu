#include "util.h"

__global__ void transpose(float* M, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y <= x) return;

    int el1 = x + n * y;
    int el2 = y + n * x;

    float temp = M[el1];
    M[el1] = M[el2];
    M[el2] = temp;
}

#define BLOCK_DIM 16

int main()
{
    SyncArray2D<float> matrix(2048);
    dim3 grid(n / BLOCK_DIM, n / BLOCK_DIM);
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    transpose<<<grid, block>>>(matrix.getDevice(), 2048);
}
