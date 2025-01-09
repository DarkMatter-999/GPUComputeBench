#define BLOCKSIZE 32

#define BM 64
#define BN 64
#define BK 8
#define TM 8

__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha, const float *A,const float *B, float beta, float *C) {

  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0};
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();

  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] = alpha * threadResults[resIdx] + beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}
void run_sgemm_1d_blocktiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, BN), CEIL_DIV(N, BN));
  dim3 blockDim((BM * BN) / TM);
  sgemm_1d_blocktiling<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}


