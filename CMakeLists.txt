cmake_minimum_required(VERSION 3.16)
project(accounters_C LANGUAGES CXX CUDA)

find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "-O3 -pthread")
set(CMAKE_CUDA_FLAGS "-O3")

set(SOURCE_DIR "accounters")
include_directories(${SOURCE_DIR})

set(SOURCES 
  "${SOURCE_DIR}/kernels.cu"
  "${SOURCE_DIR}/cuhashtable.cu"
  "${SOURCE_DIR}/cpphashtable.cpp"
  "${SOURCE_DIR}/kmers.cpp"
  #"${SOURCE_DIR}/revcomp.cpp"
)

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "${SOURCE_DIR}/bindings.cpp")
