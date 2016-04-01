#include "reverse.cuh"

#define N 1000

int main()
{
	srand(time(NULL));

	int *tab;
	int *devTab;

	cudaMallocHost((void**)&tab, sizeof(tab[0]) * N);
	cudaMalloc((void**)&devTab, sizeof(devTab[0]) * N);

	for (int i = 0; i < N; ++i) {
		tab[i] = rand();
	}
}
