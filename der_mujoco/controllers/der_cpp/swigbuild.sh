#!/bin/bash

swig -c++ -python -o Der_iso_wrap.cpp Der_iso.i
g++ -c Der_iso.cpp Der_iso_wrap.cpp Der_utils.cpp -I/home/qj/anaconda3/envs/rlenv/include/python3.6m -fPIC -std=c++14 -O2
g++ -shared Der_iso.o Der_iso_wrap.o Der_utils.o -o _Der_iso.so -fPIC
python -c "import _Der_iso"
