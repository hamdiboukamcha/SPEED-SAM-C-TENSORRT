cmake_minimum_required(VERSION 3.18)
project(nanosam LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress ZERO_CHECK generation in Visual Studio
set(CMAKE_SUPPRESS_REGENERATION ON)

# Find packages
find_package(CUDA REQUIRED)

# Set OpenCV root directory
set(OpenCV_DIR "F:/Program Files (x86)/opencv-4.10.0-windows/opencv/build")
find_package(OpenCV REQUIRED)

# Set TensorRT root directory
set(TENSORRT_ROOT "F:/Program Files/TensorRT-8.6.1.6")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS ${TENSORRT_ROOT}/include)
find_library(NVINFER_LIBRARY nvinfer PATHS ${TENSORRT_ROOT}/lib)
find_library(NVONNXPARSER_LIBRARY nvonnxparser PATHS ${TENSORRT_ROOT}/lib)

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include  # Add include directory
)

# Add source files
file(GLOB SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# Add header files
file(GLOB HEADERS
    ${CMAKE_SOURCE_DIR}/include/*.h
)

# Organize headers under "Header Files" in Visual Studio
source_group("Header Files" FILES ${HEADERS})

# CUDA compilation flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++17")

# Add main executable without generating ALL_BUILD
add_executable(SpeedSAM ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(SpeedSAM
    ${NVINFER_LIBRARY}
    ${NVONNXPARSER_LIBRARY}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)

# Set up custom CUDA runtime library if needed
set_target_properties(SpeedSAM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# TensorRT and CUDA runtime flags
target_compile_definitions(SpeedSAM PRIVATE USE_TENSORRT USE_CUDA)

# Export paths for flexibility in multi-platform use
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "TensorRT Libraries: ${TENSORRT_INCLUDE_DIR}, ${NVINFER_LIBRARY}, ${NVONNXPARSER_LIBRARY}")
message(STATUS "OpenCV Directory: ${OpenCV_DIR}")

# Set main executable as default startup project (optional)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT SpeedSAM)
