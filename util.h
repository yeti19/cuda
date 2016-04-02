/*#define CUDA_CHECK_RETURN(value) { \
cudaError_t _m_cudaStat = value; \
if (_m_cudaStat != cudaSuccess) { \
    fprintf(stderr, "Error %s at line %d in file %s\n", \
    cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
    exit(1); \
} }*/
#define CUDA_CHECK_RETURN(value)

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
class SyncMemory {
    void *host, *device;
    size_t size;
public:
    SyncMemory(int n) {
        size = sizeof(T) * n;
        cudaMallocHost((void**) &host, size);
        cudaMalloc((void**) &device, size);
    }
    ~SyncMemory() {
        cudaFreeHost((void*) host);
        cudaFree((void*) device);
    }

    void syncToDevice() {
        cudaMemcpy(host, device, size, cudaMemcpyHostToDevice);
    }
    void syncToHost() {
        cudaMemcpy(device, host, size, cudaMemcpyDeviceToHost);
    }

    T *getHost() { return static_cast<T*>(host); }
    T *getDevice() { return static_cast<T*>(device); }
};
