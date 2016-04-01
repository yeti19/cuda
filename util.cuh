#define CUDA_CHECK_RETURN(value) { \
cudaError_t _m_cudaStat = value; \
if (_m_cudaStat != cudaSuccess) { \
	fprintf(stderr, "Error %s at line %d in file %s\n", \
	cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
	exit(1); \
} }

class Timer {
	timeval start;
public:
	void startTimer() {
		gettimeofday(&start, 0);
	}

	float stopTimer() {
		timeval end;
		gettimeofday(&end, 0);
		float sec = end.tv_sec - start.tv_sec;
		float usec = end.tv_usec - start.tv_usec;
		return sec + (usec / 1000000.0);
	}
};

