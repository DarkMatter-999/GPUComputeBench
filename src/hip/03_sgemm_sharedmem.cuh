#define BLOCKSIZE 32


__global__ void sgemm_shared_mem(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float tmp = 0.0;
  for(int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
    }

    __syncthreads();

  }

  C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

void run_sgemm_shared_mem(int M, int N, int K, float alpha, float *A, float *B,
    float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  sgemm_shared_mem<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
