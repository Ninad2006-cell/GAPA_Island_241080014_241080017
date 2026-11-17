cmake_minimum_required(VERSION 3.24)
project(Parallel_GA_Research LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configure CUDA architecture (target Kaggle Tesla T4 by default: sm_75)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Find CUDA Toolkit and OpenMP
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Common CUDA Kernel Object/Static Library
add_library(fitness_kernel STATIC
    cuda/fitness_kernel.cu
)
set_target_properties(fitness_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(fitness_kernel PRIVATE CUDA::cudart)

# Helper macro for approaches
macro(add_ga_approach name source_file)
    add_executable(${name} ${source_file})
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${name} PRIVATE
        fitness_kernel
        CUDA::cudart
        OpenMP::OpenMP_CXX
    )
    # Ensure NVCC passes OpenMP flags to host compiler when compiling .cu with OpenMP
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
    )
endmacro()

# 1. Single Process GA
add_ga_approach(single approaches/single.cu)

# 2. SM GA (Single-process with parallel fitness)
add_ga_approach(sm approaches/sm.cu)

# 3. M GA (Multi-process genetic operations)
add_ga_approach(m approaches/m.cu)

# 4. MNM GA (Multi-process operations with nested parallel fitness)
add_ga_approach(mnm approaches/mnm.cu)

# 5. Proposed Island-Based GA (Batched GPU fitness across islands)
add_ga_approach(island approaches/island.cu)

# Unified Benchmark Driver Executable
set_source_files_properties(experiments/benchmark.cpp PROPERTIES LANGUAGE CUDA)
add_executable(benchmark experiments/benchmark.cpp)
set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(benchmark PRIVATE GA_NO_MAIN)
target_link_libraries(benchmark PRIVATE
    fitness_kernel
    CUDA::cudart
    OpenMP::OpenMP_CXX
)
target_compile_options(benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
)
