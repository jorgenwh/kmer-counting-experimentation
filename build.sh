#!/bin/bash

RELEASE=0

# Parse arguments and set relevant flags
for var in "$@"
do
  if [ "$var" = "-release" ];
  then
    RELEASE=1
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

function run_cmake() {
  if [ $RELEASE = 1 ]; 
  then
    cmake .. -D CMAKE_BUILD_TYPE=Release
  else
    cmake .. -D CMAKE_BUILD_TYPE=Debug
  fi
}

function run_output() {
  printf -- "\n---------- PROGRAM OUTPUT ----------\n"
  ./temp/kmer
}

# Ensure the build dir 'temp/' exists before using it to configure build
ensure_build_dir_exists
cd temp

# Configure build
run_cmake
abort_if_not_ok

# Build binaries
make
abort_if_not_ok
cd ..

# Run program
run_output
