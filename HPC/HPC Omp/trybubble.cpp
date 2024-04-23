#include <iostream>
#include <vector>
#include <cstdlib>  // For rand()
#include <ctime>    // For clock()
#include <omp.h>    // For OpenMP

// Sequential Bubble Sort
void bubbleSortSequential(int* arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Parallel Bubble Sort using OpenMP
void bubbleSortParallel(int* arr, int size) {
    #pragma omp parallel for shared(arr, size)
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    int size;
    std::cout << "Enter the size of the array: ";
    std::cin >> size;

    int arr[size];
    std::cout << "Enter " << size << " integers:" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cin >> arr[i];
    }

    // Measure sequential bubble sort time
    clock_t startSeqBubble = clock();
    bubbleSortSequential(arr, size);
    clock_t endSeqBubble = clock();
    double timeSeqBubble = double(endSeqBubble - startSeqBubble) / CLOCKS_PER_SEC;

    // Measure parallel bubble sort time
    int par_arr[size];
    std::copy(arr, arr + size, par_arr);

    clock_t startParBubble = clock();
    bubbleSortParallel(par_arr, size);
    clock_t endParBubble = clock();
    double timeParBubble = double(endParBubble - startParBubble) / CLOCKS_PER_SEC;

    // Print execution times
    std::cout << "Sequential Bubble Sort Time: " << timeSeqBubble << " seconds" << std::endl;
    std::cout << "Parallel Bubble Sort Time: " << timeParBubble << " seconds" << std::endl;

    return 0;
}


