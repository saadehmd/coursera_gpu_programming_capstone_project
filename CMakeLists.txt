cmake_minimum_required(VERSION 3.21)
project(cuda_pcl_experiments LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configure CUDA architecture (Example: 86 for Ampere, 89 for Ada Lovelace, or 'native')
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(CCCL REQUIRED CONFIG)
find_package(PCL 1.12 REQUIRED COMPONENTS common io gpu visualization segmentation)

include_directories(${PCL_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(PCL_GPU_LIBS
    pcl_gpu_octree
    pcl_gpu_containers
    pcl_gpu_segmentation
)

set(REQUIRED_LIBS
    ${PCL_LIBRARIES}
    ${PCL_GPU_LIBS}
    CCCL::CCCL
    CUDA::cudart
)

# Define target executable mapping matching files natively
set_source_files_properties(src/cuda_euclidean_clustering.cpp PROPERTIES LANGUAGE CUDA)
add_library(cuda_euclidean_clustering SHARED
    src/cuda_euclidean_clustering.cpp 
    #src/custom_kernels.cu
)
target_link_libraries(cuda_euclidean_clustering  PRIVATE ${REQUIRED_LIBS})
target_compile_options(cuda_euclidean_clustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(cuda_euclidean_clustering_main
    src/cuda_euclidean_clustering.cpp
)
target_link_libraries(cuda_euclidean_clustering_main PRIVATE 
        cuda_euclidean_clustering
        ${REQUIRED_LIBS}
)
target_compile_options(cuda_euclidean_clustering_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(create_benchmarks
    src/create_benchmarks.cpp
)
target_link_libraries(create_benchmarks PRIVATE
        cuda_euclidean_clustering
        ${REQUIRED_LIBS}
)


# Pass CCCL version to source code as preprocessor macros
target_compile_definitions(cuda_euclidean_clustering PRIVATE
    CCCL_VERSION_MAJOR=${CCCL_VERSION_MAJOR}
    CCCL_VERSION_MINOR=${CCCL_VERSION_MINOR}
)

target_compile_definitions(cuda_euclidean_clustering_main PRIVATE
    CCCL_VERSION_MAJOR=${CCCL_VERSION_MAJOR}
    CCCL_VERSION_MINOR=${CCCL_VERSION_MINOR}
)

target_compile_definitions(create_benchmarks PRIVATE
    CCCL_VERSION_MAJOR=${CCCL_VERSION_MAJOR}
    CCCL_VERSION_MINOR=${CCCL_VERSION_MINOR}
)
message(STATUS "CCCL_VERSION_MAJOR: ${CCCL_VERSION_MAJOR}")
message(STATUS "CCCL_VERSION_MINOR: ${CCCL_VERSION_MINOR}")



# Avoid internal header collision parameters
target_include_directories(cuda_euclidean_clustering PRIVATE /usr/local/cuda/include)
