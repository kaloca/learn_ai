#include <stdio.h>

__global__ void hello()
{
    printf("Hello from thread %d\n", threadIdx.x);
}

int main()
{
    hello<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
