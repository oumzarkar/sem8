#include <stdio.h>

// CUDA kernel to find the maximum value in an array
__global__ void findMaximum(int *arr, int *max, int size) {
    // Calculate thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for storing partial maximum values
    __shared__ int s_max[256];

    // Each thread loads one element into shared memory
    if (tid < size) {
        s_max[threadIdx.x] = arr[tid];
    } else {
        s_max[threadIdx.x] = INT_MIN; // Set to minimum integer value for elements outside array size
    }

    // Synchronize threads to ensure all elements are loaded
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            // Each thread compares two values and stores the maximum
            if (s_max[threadIdx.x] < s_max[threadIdx.x + s]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + s];
            }
        }
        // Synchronize threads after each reduction step
        __syncthreads();
    }

    // Store the block maximum value to global memory
    if (threadIdx.x == 0) {
        max[blockIdx.x] = s_max[0];
    }
}

int main() {
    int size = 1024; // Size of the array
    int *arr, *d_arr, *d_max, max;
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
    cudaMalloc((void**)&d_max, grid_size * sizeof(int));

    // Copy the array from host to device
    cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    findMaximum<<<grid_size, block_size>>>(d_arr, d_max, size);

    // Copy the maximum value from device to host
    cudaMemcpy(&max, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    // Find the maximum value from the maximum values of each block
    for (int i = 1; i < grid_size; i++) {
        if (max < d_max[i]) {
            max = d_max[i];
        }
    }

    // Print the maximum value
    printf("Maximum value: %d\n", max);

    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_max);

    // Free host memory
    free(arr);

    return 0;
}
