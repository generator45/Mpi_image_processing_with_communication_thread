#ifndef PCA_H
#define PCA_H

/**
 * @brief Computes top K eigenvectors from covariance matrix
 * 
 * @param covariance_matrix Input covariance matrix (num_dimensions × num_dimensions)
 * @param num_dimensions Size of the covariance matrix
 * @param K Number of principal components to extract
 * @param eigenvectors Output: K eigenvectors stored row-wise (K × num_dimensions)
 * @return 1 on success, 0 on failure
 */
int compute_top_eigenvectors(int* covariance_matrix, int num_dimensions, 
                            int K, double* eigenvectors);

/**
 * @brief Projects an image onto the top K principal components
 * 
 * @param image Centered image vector (num_dimensions)
 * @param mean Mean vector (num_dimensions)
 * @param eigenvectors K eigenvectors stored row-wise (K × num_dimensions)
 * @param num_dimensions Size of image vector
 * @param K Number of principal components
 * @param projection Output: projection coefficients (K values)
 * @return 1 on success
 */

int project(int** img_arr, double* eigenvectors, int num_dimensions, int num_components,int num_images,int *** out_img_arr);

/**
 * @brief Reconstructs an image from its PCA projection
 * 
 * @param projection Projection coefficients (K values)
 * @param mean Mean vector to add back (num_dimensions)
 * @param eigenvectors K eigenvectors stored row-wise (K × num_dimensions)
 * @param num_dimensions Size of output image
 * @param K Number of principal components
 * @param reconstructed Output: reconstructed image (num_dimensions)
 * @return 1 on success
 */

#endif // PCA_H
