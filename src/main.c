#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "config.h"
#include "timer.h"
#include "image_loader.h"
#include "statistics.h"
#include "pca.h"

typedef struct {
    double load_time;
    double local_mean_time;
    double global_mean_time;
    double local_cov_time;
    double global_cov_time;
    double eigen_time;
} TimingStats;

typedef struct {
    int** image_arr;
    int num_images;
    int* local_sum;
    int* total_sum;
    int* mean;
    int* local_cov;
    int* total_cov;
    double* eigenvectors;
    int** out_arr;
} Resources;

void print_timing_stats(int rank, TimingStats* stats) {
    if (rank == 0) {
        printf("\n=== Performance Timing Results ===\n");
        printf("Load images:        %.6f seconds\n", stats->load_time);
        printf("Local mean:         %.6f seconds\n", stats->local_mean_time);
        printf("Global mean reduce: %.6f seconds\n", stats->global_mean_time);
        printf("Local covariance:   %.6f seconds\n", stats->local_cov_time);
        printf("Global cov reduce:  %.6f seconds\n", stats->global_cov_time);
        printf("Eigenvector comp:   %.6f seconds\n", stats->eigen_time);
        printf("==================================\n\n");
    }
}
void cleanup (Resources* resources) {

    if (resources->local_sum) free(resources->local_sum);
    if (resources->total_sum) free(resources->total_sum);
    if (resources->mean) free(resources->mean);
    if (resources->local_cov) free(resources->local_cov);
    if (resources->total_cov) free(resources->total_cov);
    if (resources->eigenvectors) free(resources->eigenvectors);
    
    MPI_Finalize();
}

int main(int argc, char** argv) {
    int rank, size;
    TimingStats timing = {0};

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ===== STEP 1: Load Images =====
    int num_images, num_dimensions;
    // int** image_arr;


    Resources resources = {
        .image_arr = NULL,
        .local_sum = NULL,
        .total_sum = NULL,
        .mean = NULL,
        .local_cov = NULL,
        .total_cov = NULL,
        .eigenvectors = NULL
    };

    record_time(0);
    if (!load_image_data(size, rank, &num_images, &num_dimensions, &resources.image_arr)) {
        fprintf(stderr, "Rank %d: Failed to load image data\n", rank);
        cleanup(&resources);
        return 1;
    }
    timing.load_time = record_time(1);

    if (rank == 0) {
        printf("Loaded %d images with %d dimensions each\n", 
               TOTAL_IMAGES, num_dimensions);
    }

    // ===== STEP 2: Compute Mean =====
    // Local mean computation
    record_time(0);
    resources.local_sum = (int*)calloc(num_dimensions, sizeof(int));
    if (!resources.local_sum) {
        fprintf(stderr, "Rank %d: Failed to allocate local_sum\n", rank);
        cleanup(&resources);
        return 1;
    }

    compute_local_sum(resources.image_arr, num_images, num_dimensions, resources.local_sum);
    timing.local_mean_time = record_time(1);

    // Global mean reduction
    record_time(0);
    resources.total_sum = (int*)calloc(num_dimensions, sizeof(int));
    if (!resources.total_sum) {
        fprintf(stderr, "Rank %d: Failed to allocate total_sum\n", rank);
        cleanup(&resources);
        return 1;
    }

    MPI_Allreduce(resources.local_sum, resources.total_sum, num_dimensions, 
                  MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    timing.global_mean_time = record_time(1);

    // Compute mean by dividing sum by total images
    resources.mean = (int*)malloc(num_dimensions * sizeof(int));
    if (!resources.mean) {
        fprintf(stderr, "Rank %d: Failed to allocate mean\n", rank);
        cleanup(&resources);
        return 1;
    }

    for (int i = 0; i < num_dimensions; i++) {
        resources.mean[i] = resources.total_sum[i] / TOTAL_IMAGES;
    }

    // Center the data by subtracting mean
    center_data(resources.image_arr, resources.mean, num_images, num_dimensions);

    // ===== STEP 3: Compute Covariance Matrix =====
    // Local covariance computation
    record_time(0);
    resources.local_cov = (int*)calloc(
        (size_t)num_dimensions * num_dimensions, sizeof(int));
    if (!resources.local_cov) {
        fprintf(stderr, "Rank %d: Failed to allocate local_cov\n", rank);
        cleanup(&resources);
        return 1;
    }

    if (!compute_local_covariance(resources.local_cov, resources.image_arr, 
                                  num_images, num_dimensions)) {
        fprintf(stderr, "Rank %d: Failed to compute local covariance\n", rank);
        cleanup(&resources);
        return 1;
    }
    timing.local_cov_time = record_time(1);

    // Global covariance reduction
    record_time(0);
    resources.total_cov = (int*)malloc(
        (size_t)num_dimensions * num_dimensions * sizeof(int));
    if (!resources.total_cov) {
        fprintf(stderr, "Rank %d: Failed to allocate total_cov\n", rank);
        cleanup(&resources);
        return 1;
    }

    MPI_Allreduce(resources.local_cov, resources.total_cov, num_dimensions * num_dimensions, 
                  MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    timing.global_cov_time = record_time(1);

    // Normalize covariance by number of samples
    for (int i = 0; i < num_dimensions * num_dimensions; i++) {
        resources.total_cov[i] /= TOTAL_IMAGES;
    }

    // ===== STEP 4: Compute Eigenvectors (PCA) =====
    record_time(0);
    resources.eigenvectors = (double*)malloc(
        (size_t)num_dimensions * NUM_COMPONENTS * sizeof(double));
    if (!resources.eigenvectors) {
        fprintf(stderr, "Rank %d: Failed to allocate eigenvectors\n", rank);
        cleanup(&resources);
        return 1;
    }

    // Only rank 0 computes eigenvectors, then broadcasts
    if (rank == 0) {
        if (!compute_top_eigenvectors(resources.total_cov, num_dimensions, 
                                     NUM_COMPONENTS, resources.eigenvectors)) {
            fprintf(stderr, "Failed to compute eigenvectors\n");
            cleanup(&resources);
            return 1;
        }
    }

    // Broadcast eigenvectors to all processes
    MPI_Bcast(resources.eigenvectors, num_dimensions * NUM_COMPONENTS, 
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
    timing.eigen_time = record_time(1);

    project(resources.image_arr, resources.eigenvectors, num_dimensions, NUM_COMPONENTS, num_images, &resources.out_arr);
    // write_reconstructed_images(rank, size, num_images, num_dimensions, resources.out_arr);

    print_timing_stats(rank, &timing);
    // ===== Cleanup =====
    cleanup(&resources);
    return 0;
}
