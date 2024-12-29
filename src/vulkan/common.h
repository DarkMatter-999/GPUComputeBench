#pragma once

#include <stdlib.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define MAT_MAX 5

static void randomize_matrix(float *mat, int N) {
	for (int i = 0; i < N; i++) {
		float tmp = (float)(rand() % MAT_MAX) + 0.01 * (rand() % MAT_MAX);
		tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
		mat[i] = tmp;
	}
}
