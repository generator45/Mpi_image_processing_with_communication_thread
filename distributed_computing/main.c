#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
// 13232
#define TOTAL_IMAGES 2

int local_reduce(int** image_arr, int num_images, int image_dimension, int* local_sum) {
    for (int feature = 0; feature < image_dimension; feature++) {
        for (int img = 0; img < num_images; img++) {
            local_sum[feature] += image_arr[img][feature];
        }
    }

    return 1;
}

int load_image_data(int size, int rank, int* num_images, int* image_dimension, int*** image_arr) {

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

    *image_dimension = width * height;
    return 1;
}

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process has a 2D array of images
    int num_images, image_dimension;
    int** image_arr;
    if (!load_image_data(size, rank, &num_images, &image_dimension, &image_arr)) {
        MPI_Finalize();
        return 1;
    }

    printf("Process %d loaded %d images with dimension %d\n", rank, num_images, image_dimension);

    int* local_sum = (int*)calloc(image_dimension, sizeof(int));
    local_reduce(image_arr, num_images, image_dimension, local_sum);

    int* total_sum = (int*)calloc(image_dimension, sizeof(int));
    MPI_Allreduce(local_sum, total_sum, image_dimension, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int* mean = (int*)malloc(image_dimension * sizeof(int));

    // MPI_Barrier(MPI_COMM_WORLD);
    // printf("Process %d computed mean for each feature.\n", rank);
    for (int i = 0; i < 10; i++) {
        mean[i] = total_sum[i] / TOTAL_IMAGES;
        // printf("%d ", mean[i]);
    }
    // printf("\n");
    // MPI_Barrier(MPI_COMM_WORLD);




    MPI_Finalize();
    return 0;
}
