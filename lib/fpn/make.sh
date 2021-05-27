#!/usr/bin/env bash

cd anchors
python3 setup.py build_ext --inplace
cd ..

cd box_intersections_cpu
python3 setup.py build_ext --inplace
cd ..

cd cpu_nms
python3 build.py
cd ..

cd roi_align
python3 build.py -C src/cuda clean
python3s build.py -C src/cuda clean
cd ..

echo "Done compiling hopefully"
