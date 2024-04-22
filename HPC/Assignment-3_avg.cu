#include <stdio.h>

// CUDA kernel to find the sum of values in an array
__global__ void findSum(int *arr, int *sum, int size) {
    // Calculate thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for storing partial sum values
    __shared__ int s_sum[256];

    // Each thread loads one element into shared memory
    if (tid < size) {
        s_sum[threadIdx.x] = arr[tid];
    } else {
        s_sum[threadIdx.x] = 0; // Set to 0 for elements outside array size
    }

    // Synchronize threads to ensure all elements are loaded
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            // Each thread adds two values and stores the sum
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        // Synchronize threads after each reduction step
        __syncthreads();
    }

    // Store the block sum value to global memory
    if (threadIdx.x == 0) {
        sum[blockIdx.x] = s_sum[0];
    }
}

int main() {
    int size = 1024; // Size of the array
    int *arr, *d_arr, *d_sum, sum;
    int block_size = 256; // Number of threads per block
    int grid_size = (size + block_size - 1) / block_size; // Number of blocks

    // Allocate memory for the array on the host
    arr = (int*)malloc(size * sizeof(int));

    // Initialize array with random values
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 1000;
    }

    // Allocate memory for the array on the device
    cudaMalloc((void**)&d_arr, size * sizeof(int));
    cudaMalloc((void**)&d_sum, grid_size * sizeof(int));

    // Copy the array from host to device
    cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel to find the sum
    findSum<<<grid_size, block_size>>>(d_arr, d_sum, size);

    // Copy the sum value from device to host
    cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    // Find the total sum from the sum of each block
    for (int i = 1; i < grid_size; i++) {
        sum += d_sum[i];
    }

    // Calculate the average
    float average = (float)sum / size;

    // Print the average
    printf("Average value: %.2f\n", average);

    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_sum);

    // Free host memory
    free(arr);

    return 0;
}
