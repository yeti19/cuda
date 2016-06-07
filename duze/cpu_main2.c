#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define H(a) (-a * (log2f(a) == -INFINITY ? 0.0f : log2f(a)))
#define H2(a1, a2, sum) (H((float)(a1) / (float)(sum)) + H((float)(a2) / (float)(sum)))
#define H3(a1, a2, a3, sum) (H((float)(a1) / (float)(sum)) + H((float)(a2) / (float)(sum)) + H((float)(a3) / (float)(sum)))
#define H6(a1, a2, a3, a4, a5, a6, sum) (H3(a1, a2, a3, sum) + H3(a4, a5, a6, sum))
#define H9(a1, a2, a3, a4, a5, a6, a7, a8, a9, sum) (H3(a1, a2, a3, sum) + H3(a4, a5, a6, sum) + H3(a7, a8, a9, sum))
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
 *  - wektor wartości pierwszej zmiennej opisowej *v1s, 1 zmienna, wszystkie obiekty
 *  - wektor wartości drugiej zmiennej opisowej *v2s, 1 zmienna, wszystkie obiekty
 *  - wektor wartości zmiennych decyzyjnych *ds
 *  - ilość obiektów num_objects
 */
float compute_gig_1_2(int *v1s, int *v2s, int *ds, int num_objects)
{
    int count[2][3][3] = { 0 };

    for (int i = 0; i < num_objects; ++i) {
        int d = ds[i]; //(ds[i / 8] << (i % 8)) & 1;
        int v1 = v1s[i]; //(vars[v1_p * num_objects + i / 4] << (i % 4)) & 3;
        int v2 = v2s[i]; //(vars[v2_p * num_objects + i / 4] << (i % 4)) & 3;
        count[d][v1][v2]++;
    }

/*
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 2; ++k)
                printf("  count[%d][%d][%d] = %.0f\n", k, i, j, count[k][i][j]);

    printf("  H(decisive) = %f\n", H2(SUM_N2_N3(count, 0), SUM_N2_N3(count, 1), num_objects));
    printf("  H(v1) = %f\n", H3(SUM_N1_N3(count, 0), SUM_N1_N3(count, 1), SUM_N1_N3(count, 2), num_objects));
    printf("  H(v2) = %f\n", H3(SUM_N1_N2(count, 0), SUM_N1_N2(count, 1), SUM_N1_N2(count, 2), num_objects));
*/

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
void compute_gig_kernel(int v1_p, int v2_p, int *vars, int *ds, int num_objects, int num_vars, float *r_gig)
{
    if (v1_p >= v2_p) return;

    //printf("compute_gig(%d, %d)\n", v1_p, v2_p);
    r_gig[v1_p * num_vars + v2_p] = compute_gig_1_2(&vars[v1_p * num_objects], &vars[v2_p * num_objects], ds, num_objects);
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
    int num_objects, num_vars, *ds, *vars, result_size;
    float *gig, a_priori;
    scanf("%d %d %d %f", &num_objects, &num_vars, &result_size, &a_priori);

    vars = malloc(sizeof(int) * num_vars * num_objects);
    ds = malloc(sizeof(int) * num_objects);
    gig = malloc(sizeof(int) * num_vars * num_vars);
    for (int i = 0; i < num_objects; ++i) {
        scanf("%d", &ds[i]);
        for (int j = 0; j < num_vars; ++j)
            scanf("%d", &vars[j * num_objects + i]);
    }
    
    for (int v1_p = 0; v1_p < num_vars; ++v1_p)
        for (int v2_p = 0; v2_p < num_vars; ++v2_p)
            compute_gig_kernel(v1_p, v2_p, vars, ds, num_objects, num_vars, gig);

    struct GigStruct *gig_structs = malloc(sizeof(struct GigStruct) * num_vars * num_vars);
    int num_structs = 0;
    for (int v1_p = 0; v1_p < num_vars; ++v1_p)
        for (int v2_p = 0; v2_p < num_vars; ++v2_p) {
            gig_structs[num_structs].gig = gig[v1_p * num_vars + v2_p];
            gig_structs[num_structs].v1 = v1_p;
            gig_structs[num_structs++].v2 = v2_p;
        }

    qsort(gig_structs, num_structs, sizeof(struct GigStruct), compare_gig);

    for (int i = result_size; i >= 0; --i)
        printf("%f %d %d\n", gig_structs[i].gig, gig_structs[i].v1, gig_structs[i].v2);
    
    free(vars);
    free(ds);
    free(gig);
    return 0;
}
