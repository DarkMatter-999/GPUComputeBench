#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

// #include "01_sgemm_naive.cuh"
// #include "02_sgemm_coalesce.cuh"
// #include "03_sgemm_sharedmem.cuh"
// #include "04_sgemm_1dblocktiling.cuh"
// #include "05_sgemm_2dblocktiling.cuh"
#include "06_sgemm_vectorize.cuh"

int main() {
	int N = 6;
	int SIZE[] = {128, 256, 512, 1024, 2048, 4096};

	long m, n, k, max_size;
	max_size = SIZE[N - 1];
	printf("Max size: %ld\n", max_size);

	float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

	float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;
	float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr;

	float elapsed_time;
	hipEvent_t beg, end;
	hipCheck(hipEventCreate(&beg));
	hipCheck(hipEventCreate(&end));

	A = (float *)malloc(sizeof(float) * max_size * max_size);
	B = (float *)malloc(sizeof(float) * max_size * max_size);
	C = (float *)malloc(sizeof(float) * max_size * max_size);
	C_ref = (float *)malloc(sizeof(float) * max_size * max_size);


	randomize_matrix(A, max_size * max_size);
	randomize_matrix(B, max_size * max_size);
	randomize_matrix(C, max_size * max_size);

	hipCheck(hipMalloc(&dA, sizeof(float) * max_size * max_size));
	hipCheck(hipMalloc(&dB, sizeof(float) * max_size * max_size));
	hipCheck(hipMalloc(&dC, sizeof(float) * max_size * max_size));
	hipCheck(hipMalloc(&dC_ref, sizeof(float) * max_size * max_size));

	hipCheck(hipMemcpy(dA, A, sizeof(float) * max_size * max_size, hipMemcpyHostToDevice));
	hipCheck(hipMemcpy(dB, B, sizeof(float) * max_size * max_size, hipMemcpyHostToDevice));
	hipCheck(hipMemcpy(dC, C, sizeof(float) * max_size * max_size, hipMemcpyHostToDevice));
	hipCheck(hipMemcpy(dC_ref, C_ref, sizeof(float) * max_size * max_size, hipMemcpyHostToDevice));

	int repeat_times = 5;
	for (int s=0; s<N; s++) {
		int size = SIZE[s];
		m = n = k = size;

#ifdef DEBUG
		printf("dimensions(m=n=k) %ld, alpha: %f, beta: %f\n", m, alpha, beta);
#endif

		hipCheck(hipEventRecord(beg));

		for (int j = 0; j < repeat_times; j++) {
			run_sgemm_vectorize(m, n, k, alpha, dA, dB, beta, dC);
		}

		hipCheck(hipEventRecord(end));
		hipCheck(hipEventSynchronize(beg));
		hipCheck(hipEventSynchronize(end));
		hipCheck(hipEventElapsedTime(&elapsed_time, beg, end));
		elapsed_time /= 1000.;

		long flops = 2 * m * n * k;
		printf(
				"Average elapsed time: %fs, performance: %fGFLOPS. size: %ld\n",
				elapsed_time / repeat_times,
				(repeat_times * flops * 1e-9) / elapsed_time, m);
		fflush(stdout);

		hipCheck(hipMemcpy(dC, dC_ref, sizeof(float) * m * n, hipMemcpyDeviceToDevice));

	}

	free(A);
	free(B);
	free(C);
	free(C_ref);
	hipCheck(hipFree(dA));
	hipCheck(hipFree(dB));
	hipCheck(hipFree(dC));
	hipCheck(hipFree(dC_ref));
	return 0;
}
