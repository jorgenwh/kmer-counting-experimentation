#!/bin/bash

FASTA_FILE="data/testreads20m.fa"
CHUNK_SIZE=5000000

RELEASE=0
VALGRIND=0

# Parse arguments and set relevant flags
for var in "$@"
do
  if [ "$var" = "-r" ]; # Release build (optimizied)
  then
    RELEASE=1
  elif [ "$var" = "-v" ]; # Run valgrind 
  then
    VALGRIND=1
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
  ./temp/kmer $FASTA_FILE
}

function run_valgrind() {
  printf -- "\n---------- PROGRAM OUTPUT ----------\n"
  valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./temp/kmer $FASTA_FILE
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
make
abort_if_not_ok
cd ..

if [ $VALGRIND = 1 ];
then
  run_valgrind
  exit 0
fi

# Run program
run_output
