#!/bin/bash

DEBUG=0

# Use all cores available on the system for faster compilation
NUM_CORES=$(grep -c ^processor /proc/cpuinfo)

# Parse arguments and set relevant flags
for var in "$@" 
do
  if [ "$var" = "-d" ]; then # Debug build
    DEBUG=1
  fi
done

# Create build dir if it doesn't exist
if [ ! -d build/ ]; then
  mkdir build/
fi


# Create build
cd build 

CMAKE_ARGS=""

if [ $DEBUG = 1 ]; then
  CMAKE_ARGS+="-D CMAKE_BUILD_TYPE=Debug"
else
  CMAKE_ARGS+="-D CMAKE_BUILD_TYPE=Release"
fi

cmake .. $CMAKE_ARGS

if [ ! $? = 0 ]; then
  cd ..
  exit $?
fi


# Build
make -j${NUM_CORES}
cd ..
if [ ! $? = 0 ]; then
  exit $?
fi

