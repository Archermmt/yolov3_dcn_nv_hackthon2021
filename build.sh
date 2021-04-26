#!/usr/bin/env bash

#install paddle
echo "Installing paddle"
pip install paddlepaddle-gpu==2.0.2.post110 -f https://paddlepaddle.org.cn/whl/mkl/stable.html

#cinstall ops lis
echo "Installing ops lib"
cd ops_lib && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/quake ../ && make -j40 install
