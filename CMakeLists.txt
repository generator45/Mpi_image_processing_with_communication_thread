cmake_minimum_required(VERSION 3.28.3)
project(openMpiProj C)

set(CMAKE_C_STANDARD 11)

# Output directory
set(bin "bin/")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${bin})

# Find MPI package
find_package(MPI REQUIRED)

# Library names
set(LAPACKE_LIB lapacke)
set(LAPACK_LIB lapack)
set(BLAS_LIB blas)
set(MATH_LIB m)

# OpenMPI paths (adjust these for your system if needed)
set(openMpi_include "/usr/lib/x86_64-linux-gnu/openmpi/include")
set(openMpi_openMpi_include "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/")
set(openMpi_lib_dir "/usr/lib/x86_64-linux-gnu/openmpi/lib")
set(openMpi_Lib mpi)

# Include directories
include_directories(
    ${openMpi_include}
    ${openMpi_openMpi_include}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/
)

# Link directories
link_directories(${openMpi_lib_dir})

# Common libraries for all targets
set(COMMON_LIBS
    ${openMpi_Lib}
    ${LAPACKE_LIB}
    ${LAPACK_LIB}
    ${BLAS_LIB}
    ${MATH_LIB}
)

# ============================================
# Main PCA Application (Refactored)
# ============================================
set(PCA_SOURCES
    src/main.c
    src/image_loader.c
    src/statistics.c
    src/pca.c
    src/timer.c
)

set(PCA_HEADERS
    src/config.h
    src/image_loader.h
    src/statistics.h
    src/pca.h
    src/timer.h
    src/stb_image.h
    src/stb_image_write.h
)

add_executable(pca_mpi ${PCA_SOURCES} ${PCA_HEADERS})
target_link_libraries(pca_mpi ${COMMON_LIBS})

add_executable(original "./distributed_computing/main.c")
target_link_libraries(original ${COMMON_LIBS})
