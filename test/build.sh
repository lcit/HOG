#!/bin/bash
rm -rdf ./build > /dev/null 2>&1
rm output* > /dev/null 2>&1
mkdir build 
cd build
cmake ..
make
cd ..
