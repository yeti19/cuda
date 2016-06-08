#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define H(a) (-a * log2f(a))
#define H2(a1, a2, p) (H(((float)(a1) + (p)) / ((float)(a1 + a2) + 1.0f)) + \
                       H(((float)(a2) + (1.0f - p)) / ((float)(a1 + a2) + 1.0f)))

/* Makra do sumowania tablicy 2 x 3 x 3 */
#define SUM_N3(a, n1, n2) (a[n1][n2][0] + a[n1][n2][1] + a[n1][n2][2])
#define SUM_N2(a, n1, n3) (a[n1][0][n3] + a[n1][1][n3] + a[n1][2][n3])
#define SUM_N1(a, n2, n3) (a[0][n2][n3] + a[1][n2][n3])

#define SUM_N2_N3(a, n1) (SUM_N3(a, n1, 0) + SUM_N3(a, n1, 1) + SUM_N3(a, n1, 2))
#define SUM_N1_N3(a, n2) (SUM_N3(a, 0, n2) + SUM_N3(a, 1, n2))
#define SUM_N1_N2(a, n3) (SUM_N2(a, 0, n3) + SUM_N2(a, 1, n3))

__device__ float compute_gig_1_2(int v1_p, int v2_p, char *vars, char *ds, int vars_width, int num_objects, float p)
{
    int count[2][3][3] = { 0 };

#pragma unroll 4
    for (int i = 0; i < num_objects; ++i) {
        char d = (ds[i / 8] >> (i % 8)) & 1;
        char v1 = (vars[i * vars_width + v1_p / 4] >> ((v1_p % 4) * 2)) & 3;
        char v2 = (vars[i * vars_width + v2_p / 4] >> ((v2_p % 4) * 2)) & 3;
        count[d][v1][v2]++;
    }

    float ig1, ig2, ig12, h_p;
    h_p = H2(SUM_N2_N3(count, 0), SUM_N2_N3(count, 1), p);
    ig1 = h_p - SUM_N1_N3(count, 0) * H2(SUM_N3(count, 0, 0), SUM_N3(count, 1, 0), p) -
                SUM_N1_N3(count, 1) * H2(SUM_N3(count, 0, 1), SUM_N3(count, 1, 1), p) -
                SUM_N1_N3(count, 2) * H2(SUM_N3(count, 0, 2), SUM_N3(count, 1, 2), p);
    ig2 = h_p - SUM_N1_N2(count, 0) * H2(SUM_N2(count, 0, 0), SUM_N2(count, 1, 0), p) -
                SUM_N1_N2(count, 1) * H2(SUM_N2(count, 0, 1), SUM_N2(count, 1, 1), p) -
                SUM_N1_N2(count, 2) * H2(SUM_N2(count, 0, 2), SUM_N2(count, 1, 2), p);
    ig12 = h_p - SUM_N1(count, 0, 0) * H2(count[0][0][0], count[1][0][0], p) -
                 SUM_N1(count, 1, 0) * H2(count[0][1][0], count[1][1][0], p) -
                 SUM_N1(count, 2, 0) * H2(count[0][2][0], count[1][2][0], p) -
                 SUM_N1(count, 0, 1) * H2(count[0][0][1], count[1][0][1], p) -
                 SUM_N1(count, 1, 1) * H2(count[0][1][1], count[1][1][1], p) -
                 SUM_N1(count, 2, 1) * H2(count[0][2][1], count[1][2][1], p) -
                 SUM_N1(count, 0, 2) * H2(count[0][0][2], count[1][0][2], p) -
                 SUM_N1(count, 1, 2) * H2(count[0][1][2], count[1][1][2], p) -
                 SUM_N1(count, 2, 2) * H2(count[0][2][2], count[1][2][2], p);

    return ig12 - ((ig1 > ig2) ? ig1 : ig2);
}

__device__ float compute_gig_1_2_ds(int v1_p, int v2_p, char *vars1, char *vars2, char *ds, int vars1_width, int vars2_width, int num_objects, float p)
{
    int count[2][3][3] = { 0 };

#pragma unroll 4
    for (int i = 0; i < num_objects; ++i) {
        char d = (ds[i / 8] >> (i % 8)) & 1;
        char v1 = (vars1[i * vars1_width + v1_p / 4] >> ((v1_p % 4) * 2)) & 3;
        char v2 = (vars2[i * vars2_width + v2_p / 4] >> ((v2_p % 4) * 2)) & 3;
        count[d][v1][v2]++;
    }

    float ig1, ig2, ig12, h_p;
    h_p = H2(SUM_N2_N3(count, 0), SUM_N2_N3(count, 1), p);
    ig1 = h_p - SUM_N1_N3(count, 0) * H2(SUM_N3(count, 0, 0), SUM_N3(count, 1, 0), p) -
                SUM_N1_N3(count, 1) * H2(SUM_N3(count, 0, 1), SUM_N3(count, 1, 1), p) -
                SUM_N1_N3(count, 2) * H2(SUM_N3(count, 0, 2), SUM_N3(count, 1, 2), p);
    ig2 = h_p - SUM_N1_N2(count, 0) * H2(SUM_N2(count, 0, 0), SUM_N2(count, 1, 0), p) -
                SUM_N1_N2(count, 1) * H2(SUM_N2(count, 0, 1), SUM_N2(count, 1, 1), p) -
                SUM_N1_N2(count, 2) * H2(SUM_N2(count, 0, 2), SUM_N2(count, 1, 2), p);
    ig12 = h_p - SUM_N1(count, 0, 0) * H2(count[0][0][0], count[1][0][0], p) -
                 SUM_N1(count, 1, 0) * H2(count[0][1][0], count[1][1][0], p) -
                 SUM_N1(count, 2, 0) * H2(count[0][2][0], count[1][2][0], p) -
                 SUM_N1(count, 0, 1) * H2(count[0][0][1], count[1][0][1], p) -
                 SUM_N1(count, 1, 1) * H2(count[0][1][1], count[1][1][1], p) -
                 SUM_N1(count, 2, 1) * H2(count[0][2][1], count[1][2][1], p) -
                 SUM_N1(count, 0, 2) * H2(count[0][0][2], count[1][0][2], p) -
                 SUM_N1(count, 1, 2) * H2(count[0][1][2], count[1][1][2], p) -
                 SUM_N1(count, 2, 2) * H2(count[0][2][2], count[1][2][2], p);

    return ig12 - ((ig1 > ig2) ? ig1 : ig2);
}

__global__ void compute_gig_kernel(char *vars, char *ds, int num_objects, int num_vars, float *r_gig, float p)
{
    int v1_p = blockIdx.x * blockDim.x + threadIdx.x;
    int v2_p = blockIdx.y * blockDim.y + threadIdx.y;

    if (v1_p >= v2_p) return;
    if (v1_p >= num_vars) return;
    if (v2_p >= num_vars) return;

    const int num_v_padded = padToMultipleOf(num_vars, 32) / 4;

    r_gig[v1_p * num_vars + v2_p] = compute_gig_1_2(v1_p, v2_p, vars, ds, num_v_padded, num_objects, p);
}

struct GigStruct {
    float gig;
    int v1, v2;
};

__global__ void compute_gig_wt_kernel(char *vars, char *ds, int num_objects, int num_vars,
                                      struct GigStruct *r_gig, int max_num_gig_structs, int* num_gig_structs,
                                      float p, float threshold)
{
    if (blockIdx.x * blockDim.x >= (blockIdx.y + 1) * blockDim.y - 1) return;

    int v1_p = blockIdx.x * blockDim.x + threadIdx.x;
    int v2_p = blockIdx.y * blockDim.y + threadIdx.y;

    const int num_v_padded = padToMultipleOf(num_vars, 32) / 4;
    const int thread_n = blockDim.x * threadIdx.y + threadIdx.x;

    extern __shared__ char shared[];
    const int shared_vars_width = blockDim.y / 4;
    const int shared_vars_size = shared_vars_width * num_objects;
    for (int i = thread_n; i < shared_vars_size; i += blockDim.x * blockDim.y)
        shared[i] = vars[(i / shared_vars_width) * num_v_padded + blockIdx.y * blockDim.y / 4 + (i % shared_vars_width)];

    const int ds_size = ((num_objects - 1) / 8 + 1);
    for (int i = thread_n; i < ds_size; i += blockDim.x * blockDim.y)
        shared[shared_vars_size + i] = ds[i];
    __syncthreads();

    if (v1_p >= v2_p) return;
    if (v1_p >= num_vars) return;
    if (v2_p >= num_vars) return;

    float gig = compute_gig_1_2_ds(v1_p, threadIdx.y, vars, shared, &shared[shared_vars_size], num_v_padded, shared_vars_width, num_objects, p);
    if (gig < threshold) return;
    /* atomicInc() wraps around to 0 */
    int num = atomicAdd(num_gig_structs, 1);
    if (num < max_num_gig_structs) {
        r_gig[num].gig = gig;
        r_gig[num].v1 = v1_p;
        r_gig[num].v2 = v2_p;
    }
}

/* Komparatory do sortowania _malejąco_ */
int compare_gig(const void *a, const void *b)
{
    if (((struct GigStruct*)a)->gig > ((struct GigStruct*)b)->gig) return -1;
    else if (((struct GigStruct*)a)->gig == ((struct GigStruct*)b)->gig) return 0;
    else return 1;
}

int compare_float(const void *a, const void *b)
{
    if (*((float*)a) > *((float*)b)) return -1;
    else if (*((float*)a) == *((float*)b)) return 0;
    else return 1;
}

int main()
{
    int num_objects, num_vars, result_size, real_result_size;
    float a_priori, threshold;

    float input, copy, random_trial_kernel, random_trial_copy, random_trial_process, main_kernel, main_copy, main_process, all;
    Timer timer;
    timer.start();

    scanf("%d %d %d %f", &num_objects, &num_vars, &result_size, &a_priori);

    Sync2BitArray2D vars(num_objects, padToMultipleOf(num_vars, 32));
    SyncBitArray ds(num_objects);

    /* Czytamy dane */
    {
        for (int i = 0; i < num_objects; ++i) {
            int a; scanf("%d", &a); a &= 1;
            ds.setHost(i, a);
            for (int j = 0; j < num_vars; ++j) {
                int b; scanf("%d", &b); b &= 3;
                vars.setHost(i, j, b);
            }
        }

        input = timer.lap();
    }

    /* Kopiujemy dane na kartę */
    {
        vars.syncToDevice();
        ds.syncToDevice();

        copy = timer.lap();
    }

    /* Wykonujemy zrandomizowaną próbę na pierwszym 10% zmiennych */
    {
        int random_trial_size = num_vars / 10;
        /* Alokacja pamięci na wynikowe GIG się nie udaje gdy pamięć jest > ok. 400MB.
           XXX: Tablica gig nie musiałaby być kwadratowa. */
        if (random_trial_size > 8192)
            random_trial_size = 8192;
        float percent = (float)random_trial_size / (float)num_vars ;
        SyncArray2D<float> gig(random_trial_size, random_trial_size);

        dim3 block_size(32, 32);
        dim3 grid_size(padToMultipleOf(random_trial_size, block_size.x) / block_size.x,
                       padToMultipleOf(random_trial_size, block_size.y) / block_size.y);
        compute_gig_kernel<<<grid_size, block_size>>>((char*)vars.getDevice(), (char*)ds.getDevice(),
                                                     num_objects, random_trial_size, (float*)gig.getDevice(), a_priori);
        CUDA_CALL(cudaGetLastError());
        cudaDeviceSynchronize();
        random_trial_kernel = timer.lap();

        gig.syncToHost();
        random_trial_copy = timer.lap();

        /* Przepisujemy obliczone GIG do spójnego kawałka pamięci,
           sortujemy i wybieramy odpowiedni element jako threshold */
        {
            int num_gig = 0;
            float *gig_sorted = (float*)malloc(sizeof(float) * random_trial_size * random_trial_size);
            for (int v1_p = 0; v1_p < random_trial_size; ++v1_p)
                for (int v2_p = v1_p + 1; v2_p < random_trial_size; ++v2_p)
                    gig_sorted[num_gig++] = gig.getHostEl(v1_p, v2_p);
            qsort(gig_sorted, num_gig, sizeof(float), compare_float);
            /* gig_sorted jest posortowany malejąco */
            threshold = gig_sorted[(int)((float)result_size * percent * percent)];
            free(gig_sorted);
        }

        random_trial_process = timer.lap();
    }

    /* Wykonujemy docelowe obliczenia na wszystkich zmiennych kernelem,
       który zapisuje tylko wartości większe niż threshold */
    {
        const int max_num_structs = result_size * 2;
        SyncArray<struct GigStruct> gig_structs(max_num_structs);
        SyncVar<int> num_structs;

        int y_size = 4;
        if (num_objects < 24000) y_size = 8;
        if (num_objects < 12000) y_size = 16;
        if (num_objects < 6000) y_size = 32;

        dim3 block_size(32, y_size);
        dim3 grid_size(padToMultipleOf(num_vars, block_size.x) / block_size.x,
                       padToMultipleOf(num_vars, block_size.y) / block_size.y);
        compute_gig_wt_kernel<<<grid_size, block_size, (y_size / 4 * num_objects) + ((num_objects - 1) / 8 + 1)>>>((char*)vars.getDevice(), (char*)ds.getDevice(),
                                num_objects, num_vars, (struct GigStruct*)gig_structs.getDevice(),
                                max_num_structs, num_structs.getDevice(), a_priori, threshold);
        CUDA_CALL(cudaGetLastError());
        cudaDeviceSynchronize();
        main_kernel = timer.lap();

        num_structs.syncToHost();
        gig_structs.syncToHost();
        main_copy = timer.lap();

        real_result_size = *num_structs.getHost();
        int to_sort = real_result_size > max_num_structs ? max_num_structs : real_result_size;

        qsort(gig_structs.getHost(), to_sort, sizeof(struct GigStruct), compare_gig);

        for (int i = to_sort - 1; i >= 0; --i)
            printf("%f %d %d\n", gig_structs.getHostEl(i).gig, gig_structs.getHostEl(i).v1, gig_structs.getHostEl(i).v2);

        main_process = timer.lap();
    }

    all = input + copy + random_trial_kernel + random_trial_copy + random_trial_process + main_kernel + main_copy + main_process;
    fprintf(stderr, "data: variables, objects, result_size, true result size, threshold\n");
    fprintf(stderr, "%d, %d, %d, %d, %f\n", num_vars, num_objects, result_size, real_result_size, threshold);
    fprintf(stderr, "times: input, copy, random_trial_kernel, random_trial_copy, random_trial_process, main_kernel, main_copy, main_process, all\n");
    fprintf(stderr, "%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f\n", input, copy, random_trial_kernel,
                                    random_trial_copy, random_trial_process, main_kernel, main_copy, main_process, all);
    fprintf(stderr, "%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f\n", input / all * 100.0f, copy / all * 100.0f,
              random_trial_kernel / all * 100.0f, random_trial_copy / all * 100.0f, random_trial_process / all * 100.0f,
              main_kernel / all * 100.0f, main_copy / all * 100.0f, main_process / all * 100.0f);

    return 0;
}
