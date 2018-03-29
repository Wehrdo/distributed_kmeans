#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>

#include <mpi.h>

#define MAX_NODES (100)
#define DATA_RANGE (10000)
#define STATIC_SEED (12)

/* Datatypes */
struct {
    float centroid;
    size_t n;
} typedef Cluster_t;

struct {
    long N;
    int k;
    bool seed;
    bool run;
} typedef config_t;

/* Function declarations */
void kmeans(const float* data, const size_t n, const int k, float* clusters, int* cluster_counts);
void generateData(float* data_out, const config_t config);
int compareClusters(const void* clust1, const void* clust2);
config_t getConfig(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    MPI_Init(NULL, NULL);
    int comm_sz, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    assert(comm_sz <= MAX_NODES);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    long N;
    int k;
    bool run;
    bool rand_seed;

    double start_time, end_time;

    float* orig_data;
    if (my_rank == 0) {
        printf("P = %d\n", comm_sz);
        config_t config = getConfig(argc, argv);
        N = config.N; k = config.k; run = config.run; rand_seed = config.seed;
        orig_data = malloc(N * sizeof(float));
        if (!orig_data ) {
            printf("Failed to allocate %ld MB of data\n", N * sizeof(float) / (1 << 20));
            run = false;
        }
        generateData(orig_data, config);
        start_time = MPI_Wtime();
    }
    MPI_Bcast(&N, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&run, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rand_seed, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Abort if not initialized correctly
    if (!run) {
        MPI_Finalize();
        return 1;
    }

    // Scatter data
    int n_per_node = N / comm_sz;
    int extra = N % comm_sz;
    // create send counts and displacements
    int send_counts[MAX_NODES];
    int displacements[MAX_NODES];
    int total_displ = 0;
    for (int i = 0; i < comm_sz; ++i) {
        send_counts[i] = n_per_node + (i < extra);
        displacements[i] = total_displ;
        total_displ += send_counts[i];
    }
    // TODO: Exclude master node
    float* data = malloc(send_counts[my_rank] * sizeof(float));
    MPI_Scatterv(orig_data, send_counts, displacements, MPI_FLOAT, data, send_counts[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);


    // cluster data
    float* clusters = malloc(k * sizeof(float));
    int* cluster_counts = malloc(k * sizeof(int));

    kmeans(data, send_counts[my_rank], k, clusters, cluster_counts);

    // gather cluster data
    int n_total_clusters = k * comm_sz;
    float* all_clusters = malloc(n_total_clusters * sizeof(float));
    int* all_cluster_counts = malloc(n_total_clusters * sizeof(int));
    MPI_Gather(clusters, k, MPI_FLOAT, all_clusters, k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(cluster_counts, k, MPI_INT, all_cluster_counts, k, MPI_INT, 0, MPI_COMM_WORLD);

    // Average clusters
    if (my_rank == 0) {
        // Put cluster values and counts into a single array
        Cluster_t* combined_clusters = malloc(n_total_clusters * sizeof(Cluster_t));
        for (int i = 0; i < n_total_clusters; ++i) {
            combined_clusters[i].centroid = all_clusters[i];
            combined_clusters[i].n = all_cluster_counts[i];
        }
        // sort clusters by centroid
        qsort(combined_clusters, n_total_clusters, sizeof(Cluster_t), compareClusters);

        float* global_centroids = malloc(k * sizeof(float));
        for (int i = 0; i < k; ++i) {
            double num_sum = 0;
            double denom_sum = 0;
            for (int j = 0; j < comm_sz; ++j) {
                Cluster_t global_cluster = combined_clusters[i*comm_sz + j];
                num_sum += global_cluster.centroid * global_cluster.n;
                denom_sum += global_cluster.n;
            }
            global_centroids[i] = num_sum / denom_sum;
        }
        end_time = MPI_Wtime();
        printf("Time = %lf\n", end_time - start_time);

        for (int i = 0; i < k; ++i) {
            printf("Cluster %d = %f\n", i, global_centroids[i]);
        }
        printf("\n");

        free(combined_clusters); free(global_centroids);
    }

    free(clusters); free(cluster_counts);
    free(all_clusters); free(all_cluster_counts);
    free(data);
    if (my_rank == 0) {
        free(orig_data);
    }
    return MPI_Finalize();
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

    free(cluster_assns);
    free(centroid_sums);
}

int compareClusters(const void* clust1, const void* clust2) {
    Cluster_t* cluster1 = (Cluster_t*) clust1;
    Cluster_t* cluster2 = (Cluster_t*) clust2;
    if (cluster1->centroid < cluster2->centroid) { return -1; }
    else if (cluster1->centroid > cluster2->centroid) { return 1; }
    else { return 0; }
}

void generateData(float* data_out, const config_t config) {
    if (config.seed) {
        srand(time(NULL));
    }
    else {
        srand(STATIC_SEED);
    }
    for (size_t i = 0; i < config.N; ++i) {
        data_out[i] = rand() % DATA_RANGE;
    }
}

config_t getConfig(int argc, char* argv[]) {
    config_t config;
    // get parameters
    if (argc != 4 ||
        (argc == 4 && strcmp("y", argv[3]) && strcmp("n", argv[3]))) {
        printf("Call using kmeans <N> <k> <y|n>\n\
                N: Number of points\n\
                k: Number of clusters\n\
                y|n: Yes/No of whether to use a random seed");
        config.run = false;
    }
    else {
        config.N = atol(argv[1]);
        config.k = atoi(argv[2]);
        config.seed = strcmp("y", argv[3]) == 0;
        config.run = true;
    }
    printf("N=%ld\nk=%d\nseed=%d\n", config.N, config.k, config.seed);
    return config;
}