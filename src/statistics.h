#ifndef STATISTICS_H
#define STATISTICS_H

/**
 * @brief Computes local sum of image features
 * 
 * @param image_arr Array of images
 * @param num_images Number of images
 * @param num_dimensions Features per image
 * @param local_sum Output: accumulated sum (should be pre-allocated and zeroed)
 * @return 1 on success
 */
int compute_local_sum(int** image_arr, int num_images, 
                     int num_dimensions, int* local_sum);

/**
 * @brief Centers image data by subtracting the mean
 * 
 * @param image_arr Array of images (modified in place)
 * @param mean Mean vector to subtract
 * @param num_images Number of images
 * @param num_dimensions Features per image
 * @return 1 on success
 */
int center_data(int** image_arr, int* mean, 
               int num_images, int num_dimensions);

/**
 * @brief Computes local covariance matrix contribution
 * 
 * Uses matrix multiplication: X^T * X where X is the centered data matrix
 * 
 * @param local_cov_sum Output: accumulated covariance (should be pre-allocated and zeroed)
 * @param image_arr Array of centered images
 * @param num_images Number of images
 * @param num_dimensions Features per image
 * @return 1 on success, 0 on failure
 */
int compute_local_covariance(int* local_cov_sum, int** image_arr, 
                            int num_images, int num_dimensions);

#endif // STATISTICS_H
