#include "pca.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <lapacke.h>

int compute_top_eigenvectors(int* covariance_matrix, int num_dimensions, 
                            int K, double* eigenvectors) {
    // Convert integer covariance to double for LAPACK
    double* cov_double = (double*)malloc(
        (size_t)num_dimensions * num_dimensions * sizeof(double));
    if (!cov_double) {
        fprintf(stderr, "Failed to allocate double covariance matrix\n");
        return 0;
    }

    for (int i = 0; i < num_dimensions * num_dimensions; i++) {
        cov_double[i] = (double)covariance_matrix[i];
    }

    // Compute eigenvalues and eigenvectors using LAPACK
    double* eigenvalues = (double*)malloc(num_dimensions * sizeof(double));
    if (!eigenvalues) {
        fprintf(stderr, "Failed to allocate eigenvalues array\n");
        free(cov_double);
        return 0;
    }

    // LAPACKE_dsyev computes eigenvalues and eigenvectors
    // 'V' = compute eigenvectors, 'U' = upper triangle
    // Eigenvectors are returned in columns, sorted by eigenvalue (ascending)
    int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', 
                            num_dimensions, cov_double, 
                            num_dimensions, eigenvalues);

    if (info != 0) {
        fprintf(stderr, "LAPACK eigen decomposition failed with code %d\n", info);
        free(eigenvalues);
        free(cov_double);
        return 0;
    }

    // Extract top K eigenvectors (largest eigenvalues are at the end)
    int start_col = num_dimensions - K;
    
    for (int k = 0; k < K; k++) {
        int src_col = start_col + k;
        // Copy eigenvector column into output (stored row-wise)
        memcpy(&eigenvectors[k * num_dimensions],
               &cov_double[src_col * num_dimensions],
               num_dimensions * sizeof(double));
    }

    free(eigenvalues);
    free(cov_double);
    return 1;
}

int project(int** img_arr,double* eigenvectors, int num_dimensions, int num_components,int num_images,int *** out_img_arr) {
    int** proj = (int **)malloc(num_images * sizeof(int*));
    for (int img = 0; img < num_images; img++) {
        proj[img] = (int *)malloc(num_dimensions * sizeof(int));
        // 1. PROJECT: y = Vᵀ * (x - mean)
        // Since img_arr is already centered, x_centered = img_arr[img]
        for (int k = 0; k < num_components; k++) {
            proj[img][k] = 0.0;  // PCA coordinates
            const double* vk = &eigenvectors[k * num_dimensions]; // eigenvector k

            for (int d = 0; d < num_dimensions; d++) {
                proj[img][k] += vk[d] * (double) img_arr[img][d];
            }

            if (proj[img][k] < 0)   proj[img][k]= 0;
            if (proj[img][k] > 255) proj[img][k]= 255;
        }
    }
    *out_img_arr = proj;
    return 0;
}
