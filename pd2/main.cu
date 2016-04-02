#include "reverse.cuh"
#include "util.h"

#define N 1000

int main()
{
	srand(time(NULL));
    SyncMemory<int> tab(N);

	for (int i = 0; i < N; ++i) {
		tab.getHost()[i] = rand();
	}

    tab.syncToDevice();

    Timer t;
    t.startTimer();
    reverse<int>(tab.getDevice(), N);
    std::cout << t.stopTimer() << std::endl;
}
