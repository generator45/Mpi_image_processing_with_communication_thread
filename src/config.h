#ifndef CONFIG_H
#define CONFIG_H

// Total number of images in the dataset
#ifndef TOTAL_IMAGES
#define TOTAL_IMAGES 500
#endif

// Number of principal components to compute
#ifndef NUM_COMPONENTS
#define NUM_COMPONENTS 256
#endif

// Dataset paths
#define IMAGE_DATASET_PATH "../Datasets/lfw_processed"
#define RECONSTRUCTED_PATH "../Datasets/Reconstructed"

// Image quality for JPEG output
#define JPEG_QUALITY 90

#endif // CONFIG_H
