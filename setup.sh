#!/bin/bash
set -e
echo "Setup project"

export CC=/vpublic01/frog/wyh/gcc-6.1.0/bin/gcc
export CXX=/vpublic01/frog/wyh/gcc-6.1.0/bin/g++
export NVCC=/usr/local/cuda-9.1/bin/nvcc
rm -rf build
mkdir build
cd build/
cmake ..
make clean
make -j100 VERBOSE=1
echo "Done with setup, ready to run, call:"
echo "./mainaimGraph configuration.xml"
echo "For more information please read README.markdown"
