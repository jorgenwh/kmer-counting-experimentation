# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# compile CUDA with /usr/local/cuda-11.7/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_DEFINES = -Daccounters_C_EXPORTS

CUDA_INCLUDES = -I/home/jorgen/projects/kmer-counting-experimentation/accounters -isystem=/tmp/pip-build-env-o77kenol/overlay/include -isystem=/home/jorgen/miniconda3/include/python3.9

CUDA_FLAGS = -O3 -O3 -DNDEBUG --generate-code=arch=compute_75,code=[compute_75,sm_75] -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden -std=c++14

CXX_DEFINES = -Daccounters_C_EXPORTS

CXX_INCLUDES = -I/home/jorgen/projects/kmer-counting-experimentation/accounters -isystem /tmp/pip-build-env-o77kenol/overlay/include -isystem /home/jorgen/miniconda3/include/python3.9

CXX_FLAGS = -O3 -pthread -O3 -DNDEBUG -fPIC -fvisibility=hidden -flto -fno-fat-lto-objects

