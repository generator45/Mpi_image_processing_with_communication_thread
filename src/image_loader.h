#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

/**
 * @brief Loads image data distributed across MPI processes
 * 
 * @param size Total number of MPI processes
 * @param rank Current process rank
 * @param num_images Output: number of images assigned to this process
 * @param num_dimensions Output: total dimensions per image (width * height)
 * @param image_arr Output: 2D array of image data
 * @return 1 on success, 0 on failure
 */
int load_image_data(int size, int rank, int* num_images, 
                   int* num_dimensions, int*** image_arr);

/**
 * @brief Writes reconstructed images to disk
 * 
 * @param rank Current process rank
 * @param size Total number of MPI processes
 * @param num_images Number of images on this process
 * @param num_dimensions Total dimensions per image
 * @param image_arr Array of reconstructed image data
 * @param width Image width
 * @param height Image height
 * @param channels Number of color channels
 * @return 1 on success, 0 on failure
 */
int write_reconstructed_images(int rank, int size, int num_images, 
                               int num_dimensions, int** image_arr, 
                               int width, int height, int channels);

/**
 * @brief Reconstructs a single image and saves it to disk
 * 
 * @param image_vector Flattened image data
 * @param num_dimensions Total pixels in image
 * @param width Image width
 * @param height Image height
 * @param channels Number of color channels
 * @param filename Output filename
 * @return 1 on success, 0 on failure
 */
int reconstruct_single_image(int* image_vector, int num_dimensions, 
                            int width, int height, int channels, 
                            const char* filename);

#endif // IMAGE_LOADER_H
