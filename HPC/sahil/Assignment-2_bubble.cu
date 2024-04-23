#include <stdio.h>
#include <time.h>
#include <cuda.h>

#define N 10

__global__ void bubbleSortParallel(int *arr) {
    __shared__ int temp[N];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = 0; i < N / stride; i++) {
        temp[index] = arr[index];
        if (index + stride < N && arr[index] > arr[index + stride]) {
            temp[index + stride] = arr[index];
            arr[index] = arr[index + stride];
            arr[index + stride] = temp[index];
        }
        if (index + 1 < N && arr[index] > arr[index + 1]) {
            temp[index + 1] = arr[index];
            arr[index] = arr[index + 1];
            arr[index + 1] = temp[index];
        }
    }
}

void bubbleSortSequential(int *arr) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void printArray(int *arr) {
    for (int i = 0; i < N; i++) {
        printf("%d  ", arr[i]);
    }
    printf("\n");
}

int main() {
    int *arr, *d_arr;
    cudaEvent_t start, stop;
    float time;

    arr = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 100;
    }

    cudaMalloc((void **)&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);

    printf("Array before sorting:\n");
    printArray(arr);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bubbleSortSequential(arr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("\nArray after sequential sorting:\n");
    printArray(arr);
    printf("Sequential execution time: %f ms\n", time);


    cudaEventRecord(start);
    bubbleSortParallel<<<N/32, 32>>>(d_arr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("\nArray after parallel sorting:\n");
    printArray(arr);
    printf("Parallel execution time: %f ms\n", time);

    cudaMemcpy(arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    free(arr);

    return 0;
}
