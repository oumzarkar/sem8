#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_NODES 100
#define MAX_THREADS 4

int graph[MAX_NODES][MAX_NODES];
int visited[MAX_NODES];

void parallelBFS(int start, int n) {
    #pragma omp parallel default(none) shared(graph, visited, start, n)
    {
        #pragma omp single
        {
            visited[start] = 1;
        }

        while (1) {
            int found = 0;
            #pragma omp for
            for (int i = 0; i < n; i++) {
                if (visited[i]) {
                    #pragma omp critical
                    {
                        printf("%d is visited by thread %d\n", i, omp_get_thread_num());
                    }

                    #pragma omp for
                    for (int j = 0; j < n; j++) {
                        if (graph[i][j] && !visited[j]) {
                            #pragma omp critical
                            {
                                visited[j] = 1;
                                found = 1;
                            }
                        }
                    }
                }
            }

            #pragma omp barrier

            if (!found)
                break;
        }
    }
}

int main() {
    int n = 6; // Number of vertices

    // Initialize the graph (adjacency matrix)
    int adjacencyMatrix[6][6] = {
        {0, 1, 1, 0, 0, 0},
        {1, 0, 0, 1, 1, 0},
        {1, 0, 0, 0, 1, 0},
        {0, 1, 0, 0, 0, 1},
        {0, 1, 1, 0, 0, 1},
        {0, 0, 0, 1, 1, 0}
    };

    // Copy the adjacency matrix to the global graph variable
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            graph[i][j] = adjacencyMatrix[i][j];
        }
    }

    // Initialize visited array
    for (int i = 0; i < n; i++) {
        visited[i] = 0;
    }

    // Start BFS from vertex 0
    parallelBFS(0, n);

    return 0;
}
