cmake_minimum_required(VERSION 3.16)
project(kmer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Default to debug build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set compiler flags based on whether to perform debug or release build
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS "-g -Wall")
  set(CMAKE_CUDA_FLAGS "-arch=sm_75 -g -Wall")
else()
  set(CMAKE_CXX_FLAGS "-O3")
  set(CMAKE_CUDA_FLAGS "-arch=sm_75 -O3")
endif()

file(GLOB SOURCES
  src/dnabitset.cpp  
)

add_executable(kmer src/main.cpp ${SOURCES})
