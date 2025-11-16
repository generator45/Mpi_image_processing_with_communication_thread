#include "image_loader.h"
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "config.h"

int load_image_data(int size, int rank, int* num_images, 
                   int* num_dimensions, int*** image_arr) {
    // Calculate number of images per process
    *num_images = TOTAL_IMAGES / size;
    if (rank == size - 1) {
        *num_images += TOTAL_IMAGES % size;  // Last process takes remainder
    }

    // Allocate array of image pointers
    *image_arr = (int**)malloc(*num_images * sizeof(int*));
    if (!*image_arr) {
        fprintf(stderr, "Rank %d: Failed to allocate image array\n", rank);
        return 0;
    }

    // Calculate range of images for this process
    int start_index = rank * (TOTAL_IMAGES / size);
    int end_index = start_index + *num_images;
    
    int width = 0, height = 0, channels = 0;
    
    // Load each image assigned to this process
    for (int i = start_index; i < end_index; i++) {
        char filepath[256];
        snprintf(filepath, sizeof(filepath), "%s/%d.jpg", IMAGE_DATASET_PATH, i);

        unsigned char* img = stbi_load(filepath, &width, &height, &channels, 1);
        if (!img) {
            fprintf(stderr, "Rank %d: Failed to load image %s\n", rank, filepath);
            return 0;
        }

        // Convert to integer array
        int img_idx = i - start_index;
        (*image_arr)[img_idx] = (int*)malloc(width * height * sizeof(int));
        if (!(*image_arr)[img_idx]) {
            fprintf(stderr, "Rank %d: Failed to allocate memory for image %d\n", rank, i);
            stbi_image_free(img);
            return 0;
        }
        
        for (int j = 0; j < width * height; j++) {
            (*image_arr)[img_idx][j] = img[j];
        }
        stbi_image_free(img);
    }

    *num_dimensions = width * height;
    return 1;
}

int reconstruct_single_image(int* image_vector, int num_dimensions, 
                            int width, int height, int channels, 
                            const char* filename) {
    // Convert integer array to unsigned char
    unsigned char* img = (unsigned char*)malloc(num_dimensions * sizeof(unsigned char));
    if (!img) {
        fprintf(stderr, "Failed to allocate memory for image reconstruction\n");
        return 0;
    }
    
    for (int i = 0; i < num_dimensions; i++) {
        // Clamp values to [0, 255]
        int val = image_vector[i];
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        img[i] = (unsigned char)val;
    }
    
    int quality = 90;
    int success = stbi_write_jpg(filename, width, height, channels, img, quality);
    free(img);
    
    if (!success) {
        fprintf(stderr, "Failed to write image %s\n", filename);
        return 0;
    }
    return 1;
}

int write_reconstructed_images(int rank, int size, int num_images, 
                               int num_dimensions, int** image_arr, 
                               int width, int height, int channels) {
    if (num_images <= 0) {
        return 1;  // Nothing to write
    }

    int start_index = rank * (TOTAL_IMAGES / size);
    int end_index = start_index + num_images;

    for (int i = start_index; i < end_index; i++) {
        int img_idx = i - start_index;
        char output_path[256];
        snprintf(output_path, sizeof(output_path), 
                "%s/reconstructed_%d.jpg", RECONSTRUCTED_PATH, i);
        
        if (!reconstruct_single_image(image_arr[img_idx], num_dimensions, 
                                     width, height, channels, output_path)) {
            fprintf(stderr, "Rank %d: Failed to reconstruct image %d\n", rank, i);
            return 0;
        }
    }
    return 1;
}
