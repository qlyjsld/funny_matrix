#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 1024

double ans[N][N] __attribute__((aligned(64)));
double res[N][N] __attribute__((aligned(64)));
double mul1[N][N] __attribute__((aligned(64)));
double mul2[N][N] __attribute__((aligned(64)));

#define SM (64 / sizeof(double))

int main(void)
{
    // ... Initialize mul1 and mul2
    int i, i2, j, j2, k, k2, m, n;
    double *restrict rres;
    double *restrict rmul1;
    double *restrict rmul2;
    clock_t start, end;
    double cpu_time_used;

    srand(clock());

    printf("initializing matrix...\n\n");
    for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j) {
            mul1[i][j] = rand() % (N + 1);
            mul2[i][j] = rand() % (N + 1);
        }

    printf("brute forcing ans...\n");
    start = clock();

    for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j)
            for (k = 0; k < N; ++k)
                ans[i][j] += mul1[i][k] * mul2[k][j];

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("cpu_time_used for brute force method: %f\n\n", cpu_time_used);

    printf("solving with my solution...\n");
    start = clock();

    // simd
    // for (i = 0; i < N; i += SM)
    // 	for (j = 0; j < N; j += SM)
    // 		for (k = 0; k < N; k += SM)
    // 			for (i2 = 0, rres = &res[i][j], rmul1 = &mul1[i][k]; i2
    // < SM;
    // 			++i2, rres += N, rmul1 += N) {
    // 				_mm_prefetch (&rmul1[8], _MM_HINT_NTA);
    // 				for (k2 = 0, rmul2 = &mul2[k][j]; k2 < SM; ++k2,
    // rmul2 += N) {
    // 					__m128d m1d = _mm_load_sd (&rmul1[k2]);
    // 					m1d = _mm_unpacklo_pd (m1d, m1d);
    // 					for (j2 = 0; j2 < SM; j2 += 2) {
    // 						__m128d m2 = _mm_load_pd
    // (&rmul2[j2]);
    // 						__m128d r2 = _mm_load_pd
    // (&rres[j2]); 						_mm_store_pd
    // (&rres[j2], 						_mm_add_pd
    // (_mm_mul_pd (m2, m1d), r2));
    // 					}
    // 				}
    // 			}

    // loop reordering
    // for (i = 0; i < N; ++i)
    // 	for (k = 0; k < N; ++k)
    // 		for (j = 0; j < N; ++j)
    // 			res[i][j] += mul1[i][k] * mul2[k][j];

    // loop reordering with simd
    // for (i = 0; i < N; ++i)
    // 	for (k = 0; k < N; ++k) {
    // 		__m256d m1_v = _mm256_set1_pd(mul1[i][k]);

    // 		for (j = 0; j < N; j += SM) {
    // 			rmul2 = &mul2[k][j];
    // 			rres = &res[i][j];

    // 			for (m = 0; m < SM; m += 4) {
    // 				__m256d m2_v = _mm256_load_pd(&rmul2[m]);
    // 				__m256d res_v = _mm256_load_pd(&rres[m]);
    // 				_mm256_store_pd(&rres[m],
    // 					_mm256_add_pd(res_v, _mm256_mul_pd(m1_v,
    // m2_v)));
    // 			}
    // 		}
    // 	}

    // loop reordering with simd and unrolling
    for (i = 0; i < N; ++i)
        for (k = 0; k < N; ++k) {
            __m256d m1_v = _mm256_set1_pd(mul1[i][k]);

            for (j = 0; j < N; j += SM) {
                rmul2 = &mul2[k][j];
                rres = &res[i][j];

                __m256d m2_v = _mm256_load_pd(&rmul2[0]);
                __m256d res_v = _mm256_load_pd(&rres[0]);
                _mm256_store_pd(
                    &rres[0], _mm256_add_pd(res_v, _mm256_mul_pd(m1_v, m2_v)));

                __m256d m3_v = _mm256_load_pd(&rmul2[4]);
                __m256d res2_v = _mm256_load_pd(&rres[4]);
                _mm256_store_pd(
                    &rres[4], _mm256_add_pd(res2_v, _mm256_mul_pd(m1_v, m3_v)));
            }
        }

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("cpu_time_used for my solution: %f\n\n", cpu_time_used);

    printf("verifying...\n");
    for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j)
            if (res[i][j] != ans[i][j]) {
                printf("res is incorrect!\n");
                return 1;
            }

    printf("res is correct!\n");
    return 0;
}
