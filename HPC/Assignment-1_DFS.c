#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_NODES 100
#define MAX_THREADS 4

int graph[MAX_NODES][MAX_NODES];
int visited[MAX_NODES];

void parallelDFS(int node, int n) {
    #pragma omp parallel default(none) shared(graph, visited, node, n)
    {
        if (!visited[node]) {
            #pragma omp single
            {
                visited[node] = 1;
                printf("%d is visited by thread %d\n", node, omp_get_thread_num());
            }

            #pragma omp for
            for (int i = 0; i < n; i++) {
                if (graph[node][i] && !visited[i]) {
                    parallelDFS(i, n);
                }
            }
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

    // Start DFS from vertex 0
    parallelDFS(0, n);

    return 0;
}
