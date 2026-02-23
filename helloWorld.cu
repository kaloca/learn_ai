#include <stdio.h>

int MAX_BLOCK_SIZE = 1024;

__global__ void hello()
{
    printf("Hello from thread %d\n", threadIdx.x);
}

__global__ void vec_add(float *a, float *b, float *c, int size)
{
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i < size)
        c[i] = a[i] + b[i];
}

__global__ void mat_mul(float *A, float *B, float *C, int K, int M, int N)
{
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    float sum = 0.0;
    for (int k = 0; k < K; k++)
    {
        int row = i / N;
        int col = i % N;
        sum += A[row * K + k] * B[k * N + col];
    }
    C[i] = sum;
}

template <int T>
__global__ void tiled_mat_mul(float *A, float *B, float *C, int K, int M, int N)
{
    __shared__ float A_s[T][T];
    __shared__ float B_s[T][T];

    int x = threadIdx.x;
    int y = threadIdx.y;

    float sum = 0.0;
    for (int t = 0; t < (K + T - 1) / T; t++)
    {

        int arow = (blockIdx.y * T) + y;
        int acol = (t * T) + x;
        int brow = (t * T) + y;
        int bcol = (blockIdx.x * T) + x;

        // printf("t = %d. x = %d, y = %d. A: (%d, %d), B: (%d, %d) \n", t, x, y, arow, acol, brow, bcol);
        A_s[y][x] = (acol < K) ? A[arow * K + acol] : 0.0f;
        B_s[y][x] = (brow < K) ? B[brow * N + bcol] : 0.0f;

        __syncthreads();

        // if (threadIdx.x == 0 && threadIdx.y == 0)
        // {

        //     printf("A_s = \n[%f, %f]\n[%f, %f]\n", A_s[0][0], A_s[0][1], A_s[1][0], A_s[1][1]);
        // }

        for (int k = 0; k < T; k++)
        {
            sum += A_s[y][k] * B_s[k][x];
        }

        __syncthreads();
    }

    int crow = blockIdx.y * T + y;
    int ccol = blockIdx.x * T + x;
    if (crow < M && ccol < N)
        C[crow * N + ccol] = sum;
}

void test_vec_add()
{
    int size = 2048;

    float h_a[size];
    float h_b[size];
    float h_c[size];

    for (int i = 0; i < size; i++)
    {
        h_a[i] = i;
        h_b[i] = i * 3;
        h_c[i] = 0;
    }

    float *d_a;
    float *d_b;
    float *d_c;

    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_c, size * sizeof(float));

    cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, size * sizeof(float), cudaMemcpyHostToDevice);

    int nblocks = (size + (MAX_BLOCK_SIZE - 1)) / MAX_BLOCK_SIZE;
    int nthreads = 1024;

    printf("nblocks: %d, nthreads: %d\n", nblocks, nthreads);

    vec_add<<<nblocks, nthreads>>>(d_a, d_b, d_c, size);

    cudaDeviceSynchronize();

    float result[size];
    cudaMemcpy(result, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 6; i++)
    {
        printf("%f, ", result[i]);
    }
}

void test_mat_mul()
{
    int M = 2;
    int K = 3;
    int N = 2;

    float h_A[M * K] = {1, 2, 3, 4, 5, 6};
    float h_B[K * N] = {1, 2, 3, 4, 5, 6};
    float h_C[M * N];

    float *d_a;
    float *d_b;
    float *d_c;

    cudaMalloc(&d_a, sizeof(h_A));
    cudaMalloc(&d_b, sizeof(h_B));
    cudaMalloc(&d_c, sizeof(h_C));

    cudaMemcpy(d_a, h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_B, sizeof(h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_C, sizeof(h_C), cudaMemcpyHostToDevice);

    int nblocks = 1;      //(size + (MAX_BLOCK_SIZE - 1)) / MAX_BLOCK_SIZE;
    int nthreads = M * N; // 1024;

    printf("nblocks: %d, nthreads: %d\n", nblocks, nthreads);

    mat_mul<<<nblocks, nthreads>>>(d_a, d_b, d_c, K, M, N);

    cudaDeviceSynchronize();

    float result[M * N];
    cudaMemcpy(result, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < sizeof(result) / sizeof(float); i++)
    {
        printf("%f, ", result[i]);
    }
}

void test_tiled_mat_mul()
{
    int M = 5;
    int K = 4;
    int N = 3;
    int T = 2;

    float h_A[M * K];
    float h_B[K * N];

    for (int i = 0; i < M * K; i++)
        h_A[i] = i;
    for (int i = 0; i < K * N; i++)
        h_B[i] = i;

    float h_C[M * N];

    float *d_a;
    float *d_b;
    float *d_c;

    cudaMalloc(&d_a, sizeof(h_A));
    cudaMalloc(&d_b, sizeof(h_B));
    cudaMalloc(&d_c, sizeof(h_C));

    cudaMemcpy(d_a, h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_B, sizeof(h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_C, sizeof(h_C), cudaMemcpyHostToDevice);

    int nblocks = 1;      //(size + (MAX_BLOCK_SIZE - 1)) / MAX_BLOCK_SIZE;
    int nthreads = M * N; // 1024;

    printf("nblocks: %d, nthreads: %d\n", nblocks, nthreads);

    dim3 blockDim(T, T);
    dim3 gridDim((N + T - 1) / T, (M + T - 1) / T);
    tiled_mat_mul<2><<<gridDim, blockDim>>>(d_a, d_b, d_c, K, M, N);

    cudaDeviceSynchronize();

    float result[M * N];
    cudaMemcpy(result, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < sizeof(result) / sizeof(float); i++)
    {
        printf("%f, ", result[i]);
    }
}

int main()
{
    test_tiled_mat_mul();
    return 0;
}

// cudaMalloc(&ptr, size_in_bytes);
// cudaMemcpy(dst, src, size_in_bytes, cudaMemcpyHostToDevice);