#pragma once

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define hipCheck(ans) { hipAssert((ans), __FILE__, __LINE__); }
inline void hipAssert(hipError_t code, const char *file, int line, bool abort = true) {
	if (code != hipSuccess) {
		printf("HIP error: %s in file %s at line %d\n", hipGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void randomize_matrix(float *mat, int N) {
	for (int i = 0; i < N; i++) {
		float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
		tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
		mat[i] = tmp;
	}
}

// #define DEBUG
