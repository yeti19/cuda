#!/bin/sh

module load cuda

cd ~/kr359582/cuda/pd1
make >> build.log
./test.sh ./bin ~/tests/1pd/
cd ..
