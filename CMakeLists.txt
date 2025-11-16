set(bin "bin/")
find_package(MPI REQUIRED)
set(LAPACKE_LIB lapacke)
set(LAPACK_LIB lapack)
set(BLAS_LIB blas)
set(openMpi_include "/usr/lib/x86_64-linux-gnu/openmpi/include") 
set(openMpi_openMpi_include "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/") 
set(openMpi_lib_dir "/usr/lib/x86_64-linux-gnu/openmpi/lib")
set(openMpi_Lib mpi)
set(math_Lib m)

# find_library(LAPACKE_LIBRARY lapacke REQUIRED)
# find_library(LAPACK_LIBRARY lapack REQUIRED)
# find_library(OPENBLAS_LIBRARY openblas REQUIRED)


cmake_minimum_required(VERSION 3.28.3)
project(openMpiProj)
set(CMAKE_C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${bin})
include_directories(${openMpi_include} ${openMpi_openMpi_include}) 
link_directories(${openMpi_lib_dir})

add_executable("try" "./samples/try.c")
target_link_libraries(try
  ${openMpi_Lib}
  ${LAPACKE_LIB}
  ${LAPACK_LIB}
  ${BLAS_LIB}
  ${math_Lib}
)

add_executable("eigen" "./samples/eigen.c")
target_link_libraries(eigen
  ${openMpi_Lib}
  ${LAPACKE_LIB}
  ${LAPACK_LIB}
  ${BLAS_LIB}
  ${math_Lib}
)

add_executable("metrics" "./samples/metrics.c")
target_link_libraries(metrics
  ${openMpi_Lib}
  ${LAPACKE_LIB}
  ${LAPACK_LIB}
  ${BLAS_LIB}
  ${math_Lib}
)

add_executable("a" "./distributed_computing/main.c")
target_link_libraries(a
  ${openMpi_Lib}
  ${LAPACKE_LIB}
  ${LAPACK_LIB}
  ${BLAS_LIB}
  ${math_Lib}
)








