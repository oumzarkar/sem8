#include <stdio.h>

// CUDA kernel to find the minimum value in an array
__global__ void findMinimum(int *arr, int *min, int size) {
    // Calculate thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for storing partial minimum values
    __shared__ int s_min[256];

    // Each thread loads one element into shared memory
    if (tid < size) {
        s_min[threadIdx.x] = arr[tid];
    } else {
        s_min[threadIdx.x] = INT_MAX; // Set to maximum integer value for elements outside array size
    }

    // Synchronize threads to ensure all elements are loaded
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            // Each thread compares two values and stores the minimum
            if (s_min[threadIdx.x] > s_min[threadIdx.x + s]) {
                s_min[threadIdx.x] = s_min[threadIdx.x + s];
            }
        }
        // Synchronize threads after each reduction step
        __syncthreads();
    }

    // Store the block minimum value to global memory
    if (threadIdx.x == 0) {
        min[blockIdx.x] = s_min[0];
    }
}

int main() {
    int size = 1024; // Size of the array
    int *arr, *d_arr, *d_min, min;
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
    cudaMalloc((void**)&d_min, grid_size * sizeof(int));

    // Copy the array from host to device
    cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    findMinimum<<<grid_size, block_size>>>(d_arr, d_min, size);

    // Copy the minimum value from device to host
    cudaMemcpy(&min, d_min, sizeof(int), cudaMemcpyDeviceToHost);

    // Find the minimum value from the minimum values of each block
    for (int i = 1; i < grid_size; i++) {
        if (min > d_min[i]) {
            min = d_min[i];
        }
    }

    // Print the minimum value
    printf("Minimum value: %d\n", min);

    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_min);

    // Free host memory
    free(arr);

    return 0;
}
