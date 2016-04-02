#include<stdio.h>
#include<sys/time.h>

#define GRID_SIZE 8192
#define BLOCK_SIZE 256

__global__ void vectorAdd(int n, int* a, int* b, int* c)
{
	for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n;
		i += GRID_SIZE * BLOCK_SIZE) {

		c[i] = a[i] + b[i];
	}
}

int main() {
	int n, t;
	scanf("%d %d", &n, &t);

	int* a;
	int* devA;
	CUDA_CHECK_RETURN(cudaMallocHost((void**) &a, n * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &devA, n * sizeof(int)));

	int* b;
	int* devB;
	CUDA_CHECK_RETURN(cudaMallocHost((void**) &b, n * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &devB, n * sizeof(int)));

	int* c;
	int* devC;
	CUDA_CHECK_RETURN(cudaMallocHost((void**) &c, n * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &devC, n * sizeof(int)));

	for (int i = 0; i < n; i++) {
		a[i] = i;
		b[i] = n - i;
	}

	Timer t;
	t.startTimer();

	CUDA_CHECK_RETURN(cudaMemcpy(devA, a, sizeof(int) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devB, b, sizeof(int) * n, cudaMemcpyHostToDevice));

	vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(n, devA, devB, devC);
	CUDA_CHECK_RETURN(cudaGetLastError());

	CUDA_CHECK_RETURN(cudaMemcpy(c, devC, sizeof(int) * n, cudaMemcpyDeviceToHost));

	printf("time: %.4f\n", t.stopTimer());

	for (int i = 0; i < n; i++) {
		if (c[i] != n) {
			printf("ERROR: c[%d] = %d\n", i, c[i]);
		}
	}

	CUDA_CHECK_RETURN(cudaFreeHost((void*) a));
	CUDA_CHECK_RETURN(cudaFree((void*) devA));
	CUDA_CHECK_RETURN(cudaFreeHost((void*) b));
	CUDA_CHECK_RETURN(cudaFree((void*) devB));
	CUDA_CHECK_RETURN(cudaFreeHost((void*) c));
	CUDA_CHECK_RETURN(cudaFree((void*) devC));

	return 0;
}

