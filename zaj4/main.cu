#include "util.h"

#define ITER 256

__global__ void mandelbrot(int *out, float x_min, float y_min, float x_max, float y_max)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    float a = 0.0f, b = 0.0f;
    float a0 = x_min + (x_max - x_min) * (float)x;
    float b0 = y_min + (y_max - y_min) * (float)y;

    int i;
    for (i = 0; a*a + b*b < 4 && i < ITER; ++i) {
        float temp_a = a*a - b*b + a0;
        float temp_b = 2*a*b + b0;
        a = temp_a;
        b = temp_b;
    }

    out[x + y * blockDim.x * gridDim.x];
}

int main()
{
    syncArray2D<int> result(1024);
    mandelbrot<<<grid, block>>>(result, -2.5f, -1.0f, 1.0f, 1.0f);
}
