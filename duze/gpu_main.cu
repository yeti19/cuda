#include "util.h"

#define H(a) (-a * logf(a))
#define H2(a1, a2, sum) (H((a1) / (sum)) + H((a2) / (sum)))
#define H3(a1, a2, a3, sum) (H((a1) / (sum)) + H((a2) / (sum)) + H((a3) / (sum)))
#define H6(a1, a2, a3, a4, a5, a6, sum) (H((a1) / (sum)) + H((a2) / (sum)) + \
                                         H((a3) / (sum)) + H((a4) / (sum)) + \
                                         H((a5) / (sum)) + H((a6) / (sum)))
#define H9(a1, a2, a3, a4, a5, a6, a7, a8, a9, sum) \
                        (H3(a1, a2, a3, sum) + H3(a4, a5, a6, sum) + H3(a7, a8, a9, sum))
#define H18(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, sum) \
                                        (H6(a1, a2, a3, a4, a5, a6, sum) + \
                                         H6(a7, a8, a9, a10, a11, a12, sum) + \
                                         H6(a13, a14, a15, a16, a17, a18, sum))

/* Makra do sumowania tablicy 2 x 3 x 3 */
#define SUM_N3(a, n1, n2) (a[n1][n2][0] + a[n1][n2][1] + a[n1][n2][2])
#define SUM_N2(a, n1, n3) (a[n1][0][n3] + a[n1][1][n3] + a[n1][2][n3])
#define SUM_N1(a, n2, n3) (a[0][n2][n3] + a[1][n2][n3])

#define SUM_N2_N3(a, n1) (SUM_N3(a, n1, 0) + SUM_N3(a, n1, 1) + SUM_N3(a, n1, 2))
#define SUM_N1_N3(a, n2) (SUM_N3(a, 0, n2) + SUM_N3(a, 1, n2))
#define SUM_N1_N2(a, n3) (SUM_N2(a, 0, n3) + SUM_N2(a, 1, n3))

/* Format danych:
 *  - macierz wartości zmiennych opisowych *vars, 1 wiersz - 1 zmienna
 *  - wektor wartości zmiennych decyzyjnych *ds
 *  - ilość obiektów num_objects
 *  - wynikowe GIG
 */
__global__ void compute_gig(int *vars, int *ds, int num_objects, float *r_gig)
{
    int count[2][3][3];
    int v1_p = blockIdx.x * blockDim.x + threadIdx.x;
    int v2_p = blockIdx.y * blockDim.y + threadIdx.y;

    if (v1_p <= v2_p) return;

    for (int i = 0; i < num_objects; ++i) {
        int d = ds[i]; //(ds[i / 8] << (i % 8)) & 1;
        int v1 = vars[v1_p * num_objects + i]; //(vars[v1_p * num_objects + i / 4] << (i % 4)) & 3;
        int v2 = vars[v2_p * num_objects + i]; //(vars[v2_p * num_objects + i / 4] << (i % 4)) & 3;
        count[d][v1][v2]++;
    }

    float ig1, ig2, ig12;
    ig1 = H2(SUM_N2_N3(count, 0), SUM_N2_N3(count, 1), num_objects) +
          H3(SUM_N1_N3(count, 0), SUM_N1_N3(count, 1), SUM_N1_N3(count, 2), num_objects) -
          H6(SUM_N3(count, 0, 0), SUM_N3(count, 0, 1), SUM_N3(count, 0, 2),
             SUM_N3(count, 1, 0), SUM_N3(count, 1, 1), SUM_N3(count, 1, 2), num_objects);
    ig2 = H2(SUM_N2_N3(count, 0), SUM_N2_N3(count, 1), num_objects) +
          H3(SUM_N1_N2(count, 0), SUM_N1_N2(count, 1), SUM_N1_N2(count, 2), num_objects) -
          H6(SUM_N2(count, 0, 0), SUM_N2(count, 0, 1), SUM_N2(count, 0, 2),
             SUM_N2(count, 1, 0), SUM_N2(count, 1, 1), SUM_N2(count, 1, 2), num_objects);
    ig12 = H2(SUM_N2_N3(count, 0), SUM_N2_N3(count, 1), num_objects) +
           H9(SUM_N1(count, 0, 0), SUM_N1(count, 0, 1), SUM_N1(count, 0, 2),
              SUM_N1(count, 1, 0), SUM_N1(count, 1, 1), SUM_N1(count, 1, 2),
              SUM_N1(count, 2, 0), SUM_N1(count, 2, 1), SUM_N1(count, 2, 2), num_objects) -
       H18(count[0][0][0], count[0][0][1], count[0][0][2], count[0][1][0], count[0][1][1], count[0][1][2],
           count[0][2][0], count[0][2][1], count[0][2][2], count[1][0][0], count[1][0][1], count[1][0][2],
           count[1][1][0], count[1][1][1], count[1][1][2], count[1][2][0], count[1][2][1], count[1][2][2],
           num_objects);

    r_gig[v1_p * num_objects + v2_p] = ig12 - ((ig1 > ig2) ? ig1 : ig2);
}

int main()
{
    int num_objects, num_vars, *ds, *vars;
    scanf("%d %d", &num_objects, &num_vars);

    malloc(...);
    for (int i = 0; i < n; ++i) {
        scanf("%d", &ds[i]);
        for (int j = 0; j < k; ++j)
            scanf("%d", vars[i * num_vars + j]);
    }
    
    compute_gig<<<>>>();
    
    return 0;
}
