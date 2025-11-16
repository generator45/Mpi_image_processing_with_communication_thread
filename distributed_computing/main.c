
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// 13232
#define TOTAL_IMAGES 4000

int local_reduce(int** image_arr, int num_images, int num_dimensions, int* local_sum) {
    for (int feature = 0; feature < num_dimensions; feature++) {
        for (int img = 0; img < num_images; img++) {
            local_sum[feature] += image_arr[img][feature];
        }
    }

    return 1;
}

int load_image_data(int size, int rank, int* num_images, int* num_dimensions, int*** image_arr) {

    *num_images = TOTAL_IMAGES / size;
    if (rank == size - 1) {
        *num_images += TOTAL_IMAGES % size;
    }

    *image_arr = (int**)malloc(*num_images * sizeof(int*));

    int start_index = rank * (TOTAL_IMAGES / size);
    int end_index = start_index + *num_images;
    int width, height, channels;
    for (int i = start_index; i < end_index; i++) {
        char filepath[100];
        sprintf(filepath, "../Datasets/lfw_processed/%d.jpg", i);

        unsigned char* img = stbi_load(filepath, &width, &height, &channels, 1);
        if (!img) {
            printf("Failed to load image %s\n", filepath);
            return 0;
        }

        (*image_arr)[i - start_index] = (int*)malloc(width * height * sizeof(int));
        for (int j = 0; j < width * height; j++) {
            (*image_arr)[i - start_index][j] = img[j];
        }
        stbi_image_free(img);
    }

    *num_dimensions = width * height;
    return 1;
}

int center_mean(int** image_arr, int* mean, int num_images, int num_dimensions) {
    // fprintf(stderr, "num dim %d features %d", num_images, num_dimensions);
    for (int img = 0; img < num_images; img++) {
        for (int feature = 0; feature < num_dimensions; feature++) {
            image_arr[img][feature] -= mean[feature];
        }
    }
    return 1;
}

// m1 = m1 + m2
int reduce_matrices(int dim, int* m1, int* m2) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            m1[i * dim + j] += m2[i * dim + j];
        }
    }
    return 1;
}

int create_cov(int num_dimensions, int* local_cov, int* vec) {
    for (int i = 0; i < num_dimensions; i++) {
        for (int j = 0; j < num_dimensions; j++) {
            local_cov[i * num_dimensions + j] = vec[i] * vec[j];
        }
    }
    return 1;
}

int local_covariance_reduce(int num_dimensions, int* local_cov_sum, int** image_arr, int num_images) {
    int* local_cov = (int*)malloc(num_dimensions * num_dimensions * sizeof(int));
    if (!local_cov) return 0;
    for (int img = 0; img < num_images; img++) {
        create_cov(num_dimensions, local_cov, image_arr[img]);
        reduce_matrices(num_dimensions, local_cov_sum, local_cov);
    }
    free(local_cov);
    return 1;
}


int imageReconstruction(int* image_vector, int num_dimensions, int width, int height, int channels, const char* filename) {
    // Implement the image reconstruction logic here
    int quality = 90;
    // first convert image_vector to unsigned char*
    unsigned char* img = (unsigned char*)malloc(num_dimensions * sizeof(unsigned char));
    for (int i = 0; i < num_dimensions; i++) {
        img[i] = (unsigned char)image_vector[i];
    }
    if (!stbi_write_jpg(filename, width, height, channels, img, quality)) {
        printf("Failed to write image\n");
        free(img);
        return 0; // failed
    }
    free(img);
    return 1;
}

int writingReconstructedImages(int rank, int size, int num_images, int num_dimensions, int** image_arr, int width, int height, int channels) {
    // Use provided dimensions and channels; no sample image needed

    int start_index = rank * (TOTAL_IMAGES / size);
    int end_index = start_index + num_images;
    if (num_images <= 0) return 1; // nothing to write

    for (int i = start_index; i < end_index; i++) {
        int idx = i - start_index;
        char out_name[128];
        sprintf(out_name, "../Datasets/Reconstructed/reconstructed_%d.jpg", i);
        if (!imageReconstruction(image_arr[idx], num_dimensions, width, height, channels, out_name)) {
            return 0;
        }
    }
    return 1;
}




int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process has a 2D array of images
    int num_images, num_dimensions;
    int** image_arr;

    double t_start_total = MPI_Wtime();

    if (!load_image_data(size, rank, &num_images, &num_dimensions, &image_arr)) {
        MPI_Finalize();
        return 1;
    }
    double t_end_load = MPI_Wtime();

    // printf("Process %d loaded %d images with dimension %d\n", rank, num_images, num_dimensions);
    double t_start_mean_local = MPI_Wtime();
    int* local_sum = (int*)calloc(num_dimensions, sizeof(int));
    local_reduce(image_arr, num_images, num_dimensions, local_sum);

    int* total_sum = (int*)calloc(num_dimensions, sizeof(int));
    double t_end_mean_local = MPI_Wtime();

    double t_start_mean_global = MPI_Wtime();
    MPI_Allreduce(local_sum, total_sum, num_dimensions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    double t_end_mean_global = MPI_Wtime();

    int* mean = (int*)malloc(num_dimensions * sizeof(int));
    for (int i = 0; i < 10; i++) {
        mean[i] = total_sum[i] / TOTAL_IMAGES;
    }

    center_mean(image_arr, mean, num_images, num_dimensions);

    // Allocate covariance matrix on heap using malloc (1D contiguous for MPI)
    double t_start_cov_local = MPI_Wtime(); 
    int* local_cov_sum = (int*)malloc((size_t)num_dimensions * num_dimensions * sizeof(int));
    if (!local_cov_sum) {
        fprintf(stderr, "Failed to allocate local_cov_sum\n");
        MPI_Finalize();
        return 1;
    }
    for (int i = 0; i < num_dimensions * num_dimensions; i++)
        local_cov_sum[i] = 0;

    local_covariance_reduce(num_dimensions, local_cov_sum, image_arr, num_images);
    double t_end_cov_local = MPI_Wtime();

    int* total_cov = (int*)malloc((size_t)num_dimensions * num_dimensions * sizeof(int));
    if (!total_cov) {
        fprintf(stderr, "Failed to allocate total_cov\n");
        free(local_cov_sum);
        MPI_Finalize();
        return 1;
    }

    double t_start_cov_global = MPI_Wtime();
    MPI_Allreduce(local_cov_sum, total_cov, num_dimensions * num_dimensions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    double t_end_cov_global= MPI_Wtime();

    for (int i = 0; i < num_dimensions * num_dimensions; i++) {
        total_cov[i] /= TOTAL_IMAGES;
    }
    
    if (rank == 0) {
      printf("time to load images %f \n",t_end_load - t_start_total);
      printf("time to local mean %f \n",t_end_mean_local - t_start_mean_local);
      printf("time to gloabl mean %f \n",t_end_mean_global - t_start_mean_global);
      printf("time to local cov %f \n",t_end_cov_local - t_start_cov_local);
      printf("time to gloabl cov %f \n",t_end_cov_global - t_start_cov_global);
    }

    // Cleanup
    free(local_cov_sum);
    free(total_cov);
    free(mean);
    free(local_sum);
    free(total_sum);

    MPI_Finalize();
    return 0;
}
