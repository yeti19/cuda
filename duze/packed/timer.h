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
