#ifndef __UTIL_H__
#define __UTIL_H__

#define CUDA_CALL(call) call

int padToMultipleOf(int number, int padding) {
    return ((number - 1) / padding + 1) * padding;
}

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

class Logger {
    std::ofstream logFile;
public:
    Logger(std::string filename) {
        logFile.open(filename, ios::out | ios::app);
        if (!logFile.is_open())
            throw std::exception("Unable to open log file!");
    }
    ~Logger() {
        logFile.close();
    }

    void log(std::string message) {
        logFile << message << std::endl;
    }
    void cudaLogCall(cudaError_t err) {
        if (err != cudaSuccess) {
            printf("Error %s at line %d in file %s\n",
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);
        }
    }
};

class SyncMemory {
    void *host, *device;
    const size_t size;
public:
    SyncMemory(size_t size) : size(size){
        CUDA_CALL(cudaMallocHost((void**) &host, size));
        CUDA_CALL(cudaMalloc((void**) &device, size));
    }
    ~SyncArray() {
        CUDA_CALL(cudaFreeHost((void*) host));
        CUDA_CALL(cudaFree((void*) device));
    }

    void syncToDevice() {
        CUDA_CALL(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
    }
    void syncToHost() {
        CUDA_CALL(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    }

    void *getHost() { return host; }
    void *getDevice() { return device; }
};

template <typename T>
class SyncArray : public SyncMemory {
    size_t dim;
public:
    SyncArray(size_t n) : dim(n), SyncMemory(sizeof(T) * n) { }

    T &getHostEl(int n) { return static_cast<T[]>(getHost())[n]; }
    T &getDeviceEl(int n) { return static_cast<T[]>(getDevice())[n]; }

    void print() {
        for (int i = 0; i < dim; ++i)
            std::cout << host[i] << " ";
    }
};

template <typename T>
class SyncArray2D : public SyncMemory {
    size_t dim1, dim2;
public:
    SyncArray(size_t dim1, size_t dim2) : dim1(dim1), dim2(dim2), SyncMemory(sizeof(T) * dim1 * dim2) { }

    T &getHostEl(int n, int m) { return static_cast<T[]>(getHost())[n + dim1 * m]; }
    T &getDeviceEl(int n, int m) { return static_cast<T[]>(getDevice())[n + dim1 * m]; }
};


#endif
