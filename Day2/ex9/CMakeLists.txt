cmake_minimum_required(VERSION 2.6)

project(TheGameOfLife)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

file(GLOB SOURCE "src/*.cu" "src/*.cpp")

include_directories(include ${OpenCV_INCLUDE_DIRS})

cuda_add_executable(gol ${SOURCE})
target_link_libraries(gol ${OpenCV_LIBRARIES})

