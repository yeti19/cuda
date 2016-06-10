#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdio.h>

#define CUDA_CALL(call) { \
cudaError_t _m_cudaStat = call; \
if (_m_cudaStat != cudaSuccess) { \
	fprintf(stderr, "Error %s at line %d in file %s\n", \
	cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
	exit(1); \
} }

int debug = 0;

int padToMultipleOf(int number, int padding) {
    return ((number - 1) / padding + 1) * padding;
}

#if 1
#include <sys/time.h>
class Timer {
    timeval start_time;
public:
    void start() {
        gettimeofday(&start_time, 0);
    }

    float stop() {
        timeval end_time;
        gettimeofday(&end_time, 0);
        float sec = end_time.tv_sec - start_time.tv_sec;
        float usec = end_time.tv_usec - start_time.tv_usec;
        return (sec + (usec / 1000000.0)) * 1000.0f;
    }

    float lap() {
        float time = stop();
        start();
        return time;
    }
};
#else
#include <Windows.h>
class Timer {
    LARGE_INTEGER start_time, frequency;
public:
    Timer() { QueryPerformanceFrequency(&frequency); }
    void start() {
        QueryPerformanceCounter(&start_time);
    }

    float stop() {
        LARGE_INTEGER end_time;
        QueryPerformanceCounter(&end_time);
        return (float)(end_time.QuadPart - start_time.QuadPart) * 1000000.0f / frequency.QuadPart;
    }

    float lap() {
        float time = stop();
        start();
        return time;
    }
};
#endif

class SyncMemory {
    void *host, *device;
    const size_t size;
public:
    SyncMemory(size_t size) : size(size) {
        if (debug) fprintf(stderr, "Allocating %d bytes.\n", size);
        CUDA_CALL(cudaMallocHost((void**) &host, size));
        CUDA_CALL(cudaMalloc((void**) &device, size));
    }
    ~SyncMemory() {
        if (debug) fprintf(stderr, "Freeing %d bytes.\n", size);
        CUDA_CALL(cudaFreeHost((void*) host));
        CUDA_CALL(cudaFree((void*) device));
    }

    void syncToDevice() {
        if (debug) fprintf(stderr, "Sync to device.\n", size);
        CUDA_CALL(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
    }
    void syncToHost() {
        if (debug) fprintf(stderr, "Sync to host.\n", size);
        CUDA_CALL(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    }

    void *getHost() { return host; }
    void *getDevice() { return device; }
};

template <typename T>
class SyncVar : public SyncMemory {
public:
    SyncVar() : SyncMemory(sizeof(T)) { }

    T *getHost() { return static_cast<T*>(SyncMemory::getHost()); }
    T *getDevice() { return static_cast<T*>(SyncMemory::getDevice()); }
};

template <typename T>
class SyncArray : public SyncMemory {
    size_t dim;
public:
    SyncArray(size_t n) : dim(n), SyncMemory(sizeof(T) * n) { }

    T &getHostEl(int n) { return static_cast<T*>(getHost())[n]; }
    T &getDeviceEl(int n) { return static_cast<T*>(getDevice())[n]; }
};

template <typename T>
class SyncArray2D : public SyncMemory {
    size_t dim1, dim2;
public:
    SyncArray2D(size_t dim1, size_t dim2) : dim1(dim1), dim2(dim2), SyncMemory(sizeof(T) * dim1 * dim2) { }

    T &getHostEl(int n, int m) { return static_cast<T*>(getHost())[n * dim2 + m]; }
    T &getDeviceEl(int n, int m) { return static_cast<T*>(getDevice())[n * dim2 + m]; }
};

class SyncBitArray : public SyncArray<int> {
    size_t dim;
public:
    SyncBitArray(size_t n) : dim(n), SyncArray<int>(padToMultipleOf(n, 32)) { }

    void setHost(size_t n, int a) {
        if (a == 0)
            getHostEl(n / 32) &= ~(1 << (n % 32));
        else
            getHostEl(n / 32) |= 1 << (n % 32);
    }
    int getHost(size_t n) {
        return (getHostEl(n / 32) & (1 << (n % 32))) >> (n % 32);
    }

    void print() {
        for (int i = 0; i < dim; ++i)
            printf("%d", getHost(i));
        printf("\n");
    }
};

class Sync2BitArray : public SyncArray<int> {
    size_t dim;
public:
    Sync2BitArray(size_t n) : dim(n), SyncArray<int>(padToMultipleOf(n, 16)) { }

    void setHost(size_t n, int a) {
        getHostEl(n / 16) &= ~(3 << ((n % 16) * 2));
        getHostEl(n / 16) |= a << ((n % 16) * 2);
    }
    int getHost(size_t n) {
        return (getHostEl(n / 16) & (3 << ((n % 16) * 2))) >> ((n % 16) * 2);
    }

    void print() {
        for (int i = 0; i < dim; ++i)
            printf("%d", getHost(i));
        printf("\n");
    }
};

class Sync2BitArray2D : public SyncArray2D<int> {
    size_t dim;
public:
    Sync2BitArray2D(size_t n, size_t m) : dim(m), SyncArray2D<int>(n, padToMultipleOf(m, 16)) { }

    void setHost(size_t n, size_t m, int a) {
        getHostEl(n, m / 16) &= ~(3 << ((m % 16) * 2));
        getHostEl(n, m / 16) |= a << ((m % 16) * 2);
    }
    int getHost(size_t n, size_t m) {
        return (getHostEl(n, m / 16) >> ((m % 16) * 2)) & 3;
    }
};

#endif
