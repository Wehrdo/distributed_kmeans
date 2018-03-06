#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <mpi.h>

#define K 2
#define N 10

void kmeans(const float* data, const size_t n, const int k, float* clusters, int* cluster_counts);

int main() {
    float data[N] = {3, 4, 2, 2, 4, 8, 7, 8, 9, 7};

    float clusters[K];
    int cluster_counts[K];

    kmeans(data, N, K, clusters, cluster_counts);
    for (int i = 0; i < K; ++i) {
        printf("Cluster %d = %f, n = %d\n", i, clusters[i], cluster_counts[i]);
    }
}

void kmeans(const float* data, const size_t n, const int k, float* clusters, int* cluster_counts) {
    int* cluster_assns = calloc(n, sizeof(float));
    double* centroid_sums = malloc(k * sizeof(double));
    if (!cluster_assns || !centroid_sums) {
        printf("Error! malloc failed\n");
        return;
    }

    srand(time(NULL));
    // initial cluster values
    for (size_t i = 0; i < k; ++i) {
        clusters[i] = data[rand() % n];
    }


    // whether any values switched clusters in the last iteration
    bool changed_clusters = true;
    unsigned int n_iters = 0;
    while (changed_clusters) {
        changed_clusters = false;
        // reset cluster counts and centroid sums
        for (int i = 0; i < k; ++i) {
            cluster_counts[i] = 0;
            centroid_sums[i] = 0;
        }

        for (size_t i = 0; i < n; ++i) {
            float val = data[i];
            // current distance to assigned centroid
            float closest_dist = fabs(val - clusters[cluster_assns[i]]);

            for (int cluster_i = 0; cluster_i < k; ++cluster_i) {
                float this_dist = fabs(val - clusters[cluster_i]);
                // TODO: Use epsilon instead of direct comparison?
                if (this_dist < closest_dist) {
                    closest_dist = this_dist;
                    cluster_assns[i] = cluster_i;
                    changed_clusters = true;
                }
            }

            // add value to centroid sums and increment count
            centroid_sums[cluster_assns[i]] += val;
            cluster_counts[cluster_assns[i]] += 1;
        }

        // recalculate centroids
        for (int i = 0; i < k; ++i) {
            if (cluster_counts[i] != 0) {
                clusters[i] = centroid_sums[i] / cluster_counts[i];
            }
        }
        
        n_iters++;
    }
    printf("%d iterations\n", n_iters);

    free(cluster_assns);
    free(centroid_sums);
}