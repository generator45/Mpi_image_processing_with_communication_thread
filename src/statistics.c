#include "statistics.h"
#include <stdlib.h>
#include <stdio.h>

int compute_local_sum(int** image_arr, int num_images, 
                     int num_dimensions, int* local_sum) {
    for (int img = 0; img < num_images; img++) {
        for (int feature = 0; feature < num_dimensions; feature++) {
            local_sum[feature] += image_arr[img][feature];
        }
    }
    return 1;
}

int center_data(int** image_arr, int* mean, 
               int num_images, int num_dimensions) {
    for (int img = 0; img < num_images; img++) {
        for (int feature = 0; feature < num_dimensions; feature++) {
            image_arr[img][feature] -= mean[feature];
        }
    }
    return 1;
}

int compute_local_covariance(int* local_cov_sum, int** image_arr, 
                            int num_images, int num_dimensions) {
    // Allocate transpose matrix: dimensions × images
    // int** transpose = (int**)malloc(num_dimensions * sizeof(int*));
    // if (!transpose) {
    //     fprintf(stderr, "Failed to allocate transpose matrix\n");
    //     return 0;
    // }
    //
    // for (int i = 0; i < num_dimensions; i++) {
    //     transpose[i] = (int*)malloc(num_images * sizeof(int));
    //     if (!transpose[i]) {
    //         fprintf(stderr, "Failed to allocate transpose row %d\n", i);
    //         // Free previously allocated rows
    //         for (int j = 0; j < i; j++) {
    //             free(transpose[j]);
    //         }
    //         free(transpose);
    //         return 0;
    //     }
    // }
    //
    // // Compute transpose: convert from images×dimensions to dimensions×images
    // for (int img = 0; img < num_images; img++) {
    //     for (int dim = 0; dim < num_dimensions; dim++) {
    //         transpose[dim][img] = image_arr[img][dim];
    //     }
    // }

    // Matrix multiplication: transpose * image_arr
    // Result is dimensions × dimensions covariance matrix
    // for (int i = 0; i < num_dimensions; i++) {
    //     for (int j = 0; j < num_dimensions; j++) {
    //         int sum = 0;
    //         for (int k = 0; k < num_images; k++) {
    //             // sum += transpose[i][k] * image_arr[k][j];
    //             sum += image_arr[k][i] * image_arr[k][j];
    //         }
    //         local_cov_sum[i * num_dimensions + j] += sum;
    //     }
    // }
    for (int k = 0; k < num_images; k++) {
        for (int i = 0; i < num_dimensions; i++) {
            for (int j = 0; j < num_dimensions; j++) {
                local_cov_sum[i * num_dimensions + j] += image_arr[k][i] * image_arr[k][j];
            }
        }
    }

    // Free transpose matrix
    // for (int i = 0; i < num_dimensions; i++) {
    //     free(transpose[i]);
    // }
    // free(transpose);

    return 1;
}
