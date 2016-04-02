#define CUDA_CHECK_RETURN(value) { \
cudaError_t _m_cudaStat = value; \
if (_m_cudaStat != cudaSuccess) { \
    printf("Error %s at line %d in file %s\n", \
    cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
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

template <typename T>
class SyncArray {
    T *host, *device;
    size_t dim, size;
public:
    SyncArray(size_t n) {
        size = sizeof(T) * n;
        dim = n;
        CUDA_CHECK_RETURN(cudaMallocHost((void**) &host, size));
        CUDA_CHECK_RETURN(cudaMalloc((void**) &device, size));
    }
    ~SyncArray() {
        CUDA_CHECK_RETURN(cudaFreeHost((void*) host));
        CUDA_CHECK_RETURN(cudaFree((void*) device));
    }

    void syncToDevice() {
        CUDA_CHECK_RETURN(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
    }
    void syncToHost() {
        CUDA_CHECK_RETURN(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    }

    T *getHost() { return host; }
    T *getDevice() { return device; }

    void print() {
        for (int i = 0; i < dim; ++i)
            std::cout << host[i] << " ";
    }
};
