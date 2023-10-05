#!/bin/bash

export RLENVPATH=$(which python)
RLENVPATH=$(echo "$RLENVPATH" | rev | cut -d'/' -f3- | rev)
swig -c++ -python -o Der_iso_wrap.cpp Der_iso.i
g++ -c Der_iso.cpp Der_iso_wrap.cpp Der_utils.cpp -I$HOME/eigen -I$RLENVPATH/lib/python3.8/site-packages/numpy/core/include -I$RLENVPATH/include/python3.8 -fPIC -std=c++14 -O2
g++ -shared Der_iso.o Der_iso_wrap.o Der_utils.o -o _Der_iso.so -fPIC
python -c "import _Der_iso"

