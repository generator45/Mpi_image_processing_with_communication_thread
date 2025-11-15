#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Function to load image_arr
void load_image() {
    int width, height, channels;
    // unsigned char* img = stbi_load("../Dataset/at&T/s1/1.pgm", &width, &height, &channels, 1);
    unsigned char* img = stbi_load("1.pgm", &width, &height, &channels, 1);

    // last arg = 1 => force 1 channel (grayscale)

    for(int i = 0; i < width * height; i++) {
        printf("%d ", img[i]);
    }
    if (!img) {
        printf("Failed to load image\n");
        return ;
    }

    printf("Loaded %dx%d grayscale image (channels requested = 1)\n", width, height);
}

int** load_image_arr() {
    int width, height, channels;
    // location of image --> Dataset/subfolder/imagefile.jpeg
    unsigned char* img = stbi_load("wader.jpeg", &width, &height, &channels, 1);
    // last arg = 1 => force 1 channel (grayscale)

    if (!img) {
        printf("Failed to load image\n");
        return NULL;
    }

    printf("Loaded %dx%d grayscale image (channels requested = 1)\n", width, height);

    int** image_arr = (int**)malloc(height * sizeof(int*));
    for (int i = 0; i < height; i++) {
        image_arr[i] = (int*)malloc(width * sizeof(int));
        for (int j = 0; j < width; j++) {
            image_arr[i][j] = img[i * width + j];
        }
    }

    stbi_image_free(img);
    return image_arr;
}

// Assume we got the set of all image vectors

// function to scale an image vector

// caculate the mean of all image vectors
int* calculate_mean(int** image_arr, int num_images, int image_dimension) {
    int* mean_vector = (int*)calloc(image_dimension, sizeof(int));

    for (int i = 0; i < image_dimension; i++) {
        for (int j = 0; j < num_images; j++) {
            mean_vector[i] += image_arr[j][i];
        }
    }

    for (int j = 0; j < image_dimension; j++) {
        mean_vector[j] /= num_images;
    }

    return mean_vector;
}

void center_data(int** image_arr, int num_images, int image_dimension, int* mean_vector) {
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < image_dimension; j++) {
            image_arr[i][j] -= mean_vector[j];
        }
    }
}


int main() {
    load_image();
    return 0;
}   