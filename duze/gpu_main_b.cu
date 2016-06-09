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

/* Format danych:
 *  - wektor wartości pierwszej zmiennej opisowej *v1s, 1 zmienna, wszystkie obiekty
 *  - wektor wartości drugiej zmiennej opisowej *v2s, 1 zmienna, wszystkie obiekty
 *  - wektor wartości zmiennych decyzyjnych *ds
 *  - ilość obiektów num_objects
 */
__device__ float compute_gig_1_2(int *v1s, int *v2s, int *ds, int num_objects, float p)
{
    int count[2][3][3] = { 0 };

    for (int i = 0; i < num_objects; ++i) {
        int d = ds[i]; //(ds[i / 8] << (i % 8)) & 1;
        int v1 = v1s[i]; //(vars[v1_p * num_objects + i / 4] << (i % 4)) & 3;
        int v2 = v2s[i]; //(vars[v2_p * num_objects + i / 4] << (i % 4)) & 3;
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

    //printf("  IG(v1) = %f\n", ig1);
    //printf("  IG(v2) = %f\n", ig2);
    //printf("  IG(v1 u v2) = %f\n", ig12);

    return ig12 - ((ig1 > ig2) ? ig1 : ig2);
}

/* Format danych:
 *  - macierz wartości zmiennych opisowych *vars, 1 wiersz - 1 zmienna
 *  - wektor wartości zmiennych decyzyjnych *ds
 *  - ilość obiektów num_objects
 *  - ilość zmiennych num_vars
 *  - wynikowe GIG
 */
__global__ void compute_gig_kernel(int *vars, int *ds, int num_objects, int num_vars, float *r_gig, float p)
{
    int v1_p = blockIdx.x * blockDim.x + threadIdx.x;
    int v2_p = blockIdx.y * blockDim.y + threadIdx.y;

    if (v1_p >= v2_p) return;
    if (v1_p >= num_vars) return;
    if (v2_p >= num_vars) return;
    //printf("compute_gig(%d, %d) %d\n", v1_p, v2_p, blockIdx.y);

    r_gig[v1_p * num_vars + v2_p] = compute_gig_1_2(&vars[v1_p * num_objects], &vars[v2_p * num_objects], ds, num_objects, p);
    //printf(" GIG = %f\n", r_gig[v1_p * num_vars + v2_p]);
}

struct GigStruct {
    float gig;
    int v1, v2;
};

int compare_gig(const void *a, const void *b)
{
    if (((struct GigStruct*)a)->gig > ((struct GigStruct*)b)->gig) return -1;
    else if (((struct GigStruct*)a)->gig == ((struct GigStruct*)b)->gig) return 0;
    else return 1;
}

int main()
{
    int num_objects, num_vars, result_size;
    float a_priori;
    scanf("%d %d %d %f", &num_objects, &num_vars, &result_size, &a_priori);

    SyncArray2D<int> vars(num_vars, num_objects);
    SyncArray<int> ds(num_objects);
    SyncArray2D<float> gig(num_vars, num_vars);

    for (int i = 0; i < num_objects; ++i) {
        scanf("%d", &ds.getHostEl(i));
        for (int j = 0; j < num_vars; ++j)
            scanf("%d", &vars.getHostEl(j, i));
    }

    vars.syncToDevice();
    ds.syncToDevice();

    dim3 block_size(16, 16);
    dim3 grid_size(padToMultipleOf(num_vars, block_size.x) / block_size.x,
                   padToMultipleOf(num_vars, block_size.y) / block_size.y);
    compute_gig_kernel<<<grid_size, block_size>>>((int*)vars.getDevice(), (int*)ds.getDevice(),
                                                  num_objects, num_vars, (float*)gig.getDevice(), a_priori);
    CUDA_CALL(cudaGetLastError());

    gig.syncToHost();

    struct GigStruct *gig_structs = (struct GigStruct*)malloc(sizeof(struct GigStruct) * num_vars * num_vars);
    int num_structs = 0;
    for (int v1_p = 0; v1_p < num_vars; ++v1_p)
        for (int v2_p = v1_p + 1; v2_p < num_vars; ++v2_p) {
            gig_structs[num_structs].gig = gig.getHostEl(v1_p, v2_p);
            gig_structs[num_structs].v1 = v1_p;
            gig_structs[num_structs++].v2 = v2_p;
        }

    qsort(gig_structs, num_structs, sizeof(struct GigStruct), compare_gig);

    for (int i = result_size; i >= 0; --i)
        printf("%f %d %d\n", gig_structs[i].gig, gig_structs[i].v1, gig_structs[i].v2);
    
    free(gig_structs);
    return 0;
}
