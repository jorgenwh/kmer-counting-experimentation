cmake_minimum_required(VERSION 3.16)
project(f2i LANGUAGES CXX CUDA)

set(PY_MODULE_NAME "f2i_C")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Default to debug build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set compiler flags based on whether to perform debug or release build
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS "-g -Wall")
  set(CMAKE_CUDA_FLAGS "-arch=sm_75 -g")
else()
  set(CMAKE_CXX_FLAGS "-O3")
  set(CMAKE_CUDA_FLAGS "-arch=sm_75 -O3")
endif()

# Source files used for the C++ program
set(SOURCES ${SOURCES}
  src/dnabitset.cpp  
)

# Source files used for the pybind11 module
set(PY_MODULE_SOURCES ${PY_MODULE_SOURCES}
  src/hashtable.cu
  src/kernels.cu
)

set(PY_MODULE_BINDINGS
  src/module.cpp
)

if(PY_MODULE MATCHES "True")
  # Generate build for the pybind11 module
  message(STATUS "Compiling py module: '${PY_MODULE_NAME}'")
  add_subdirectory(extern/pybind11)
  add_library(${PY_MODULE_NAME} MODULE ${PY_MODULE_SOURCES} ${PY_MODULE_BINDINGS})
  target_link_libraries(${PY_MODULE_NAME} PRIVATE pybind11::module)
  set_target_properties(${PY_MODULE_NAME} PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
  )
else()
  # Generate build for the CXX application
  message(STATUS "Compiling cxx application: '${PROJECT_NAME}'")
  add_executable(${PROJECT_NAME} src/main.cpp ${SOURCES})
endif()
