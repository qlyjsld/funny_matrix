#include <stdio.h>
#include <immintrin.h>
#define N 1024

double ans[N][N] __attribute__ ((aligned (64)));
double res[N][N] __attribute__ ((aligned (64)));
double mul1[N][N] __attribute__ ((aligned (64)));
double mul2[N][N] __attribute__ ((aligned (64)));

#define SM (64 / sizeof(double))

// ... Initialize mul1 and mul2
int i, i2, j, j2, k, k2, m, n;
double *restrict rres;
double *restrict rmul1;
double *restrict rmul2;

void brute (void)
{
    __asm volatile("# LLVM-MCA-BEGIN brute":::"memory");
	for (i = 0; i < N; ++i)
		for (j = 0; j < N; ++j)
			for (k = 0; k < N; ++k)
				res[i][j] += mul1[i][k] * mul2[k][j];
    __asm volatile("# LLVM-MCA-END":::"memory");
}

void reordering (void)
{
    __asm volatile("# LLVM-MCA-BEGIN reordering":::"memory");
	for (i = 0; i < N; ++i)
		for (k = 0; k < N; ++k)
			for (j = 0; j < N; ++j)
				res[i][j] += mul1[i][k] * mul2[k][j];
    __asm volatile("# LLVM-MCA-END":::"memory");
}

void reordering_simd (void)
{
    __asm volatile("# LLVM-MCA-BEGIN reordering_simd":::"memory");
	for (i = 0; i < N; ++i)
		for (k = 0; k < N; ++k) {
			__m256d m1_v = _mm256_set1_pd(mul1[i][k]);

			for (j = 0; j < N; j += SM) {
				rmul2 = &mul2[k][j];
				rres = &res[i][j];

				for (m = 0; m < SM; m += 4) {
					__m256d m2_v = _mm256_load_pd(&rmul2[m]);
					__m256d res_v = _mm256_load_pd(&rres[m]);
					_mm256_store_pd(&rres[m],
						_mm256_add_pd(res_v, _mm256_mul_pd(m1_v, m2_v)));
				}
			}
		}
    __asm volatile("# LLVM-MCA-END":::"memory");
}

void reordering_simd_unroll (void)
{
    __asm volatile("# LLVM-MCA-BEGIN reordering_simd_unroll":::"memory");
    for (i = 0; i < N; ++i)
	for (k = 0; k < N; ++k) {
		__m256d m1_v = _mm256_set1_pd(mul1[i][k]);

		for (j = 0; j < N; j += SM) {
			rmul2 = &mul2[k][j];
			rres = &res[i][j];

			__m256d m2_v = _mm256_load_pd(&rmul2[0]);
			__m256d res_v = _mm256_load_pd(&rres[0]);
			_mm256_store_pd(&rres[0],
				_mm256_add_pd(res_v, _mm256_mul_pd(m1_v, m2_v)));

			__m256d m3_v = _mm256_load_pd(&rmul2[4]);
			__m256d res2_v = _mm256_load_pd(&rres[4]);
			_mm256_store_pd(&rres[4],
					_mm256_add_pd(res2_v, _mm256_mul_pd(m1_v, m3_v)));
		}
	}
    __asm volatile("# LLVM-MCA-END":::"memory");
}
