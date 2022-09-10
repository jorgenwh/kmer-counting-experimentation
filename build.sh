#!/bin/bash

FASTA_FILE="data/testreads20m.fa"
CHUNK_SIZE=5000000

RELEASE=0
VALGRIND=0
PYTHON_MODULE=0
COMPILE=0

NUM_CORES=1

# Parse arguments and set relevant flags
for var in "$@" 
do
  if [ "$var" = "-r" ]; then # Release build (optimizied)
    RELEASE=1
  elif [ "$var" = "-v" ]; then # Run valgrind 
    VALGRIND=1
  elif [ "$var" = "-py" ]; then # Compile the python module
    PYTHON_MODULE=1
  elif [ "$var" = "-c" ]; then # Only compile without running the output
    COMPILE=1
  fi
done

function ensure_build_dir_exists() {
  if [ ! -d temp/ ]; then
    mkdir temp/
  fi
}

function abort_if_not_ok() {
  if [ ! $? = 0 ]; then
    cd ..
    exit $?
  fi
}

function get_num_cores() {
  NUM_CORES=$(grep -c ^processor /proc/cpuinfo)
}

function run_cmake() {
  CMAKE_ARGS=""

  if [ $RELEASE = 1 ]; then
    CMAKE_ARGS+="-D CMAKE_BUILD_TYPE=Release"
  else
    CMAKE_ARGS+="-D CMAKE_BUILD_TYPE=Debug"
  fi

  if [ $PYTHON_MODULE = 1 ]; then
    CMAKE_ARGS+=" -D PY_MODULE=True"
  else
    CMAKE_ARGS+=" -D PY_MODULE=False"
  fi

  cmake .. $CMAKE_ARGS
}

function run_output() {
  printf -- "\n---------- PROGRAM OUTPUT ----------\n"
  ./temp/f2i $FASTA_FILE
}

function run_python() {
  printf -- "\n---------- PROGRAM OUTPUT ----------\n"
  python main.py
}

function run_valgrind() {
  printf -- "\n---------- PROGRAM OUTPUT ----------\n"

  valgrind \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --verbose \
    --log-file=valgrind-out.txt \
    ./temp/f2i $FASTA_FILE

  printf -- "\n---------- VALGRIND OUTPUT ----------\n"
  cat valgrind-out.txt
  rm valgrind-out.txt
}

# Ensure the build dir 'temp/' exists before using it to configure build
ensure_build_dir_exists
cd temp

# Configure build
run_cmake
abort_if_not_ok

# Build binaries
get_num_cores
make -j${NUM_CORES}
abort_if_not_ok
cd ..

# Run valgrind
if [ $VALGRIND = 1 ]; then
  if [ $PYTHON_MODULE = 1 ]; then
    printf "\33[91mWARNING\33[0m: Cannot run valgrind on python module.\n"
    exit 0
  fi
  run_valgrind
  exit 0
fi

# Run program or py script 
if [ $COMPILE = 0 ]; then
  if [ $PYTHON_MODULE = 1 ]; then
    run_python
  else
    run_output
  fi
  exit 0
fi
