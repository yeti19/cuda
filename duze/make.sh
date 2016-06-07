
gcc cpu_main2.c -o cpu_main -std=c99 -lm
gcc cpu_main2_b.c -o cpu_main_b -std=c99 -lm
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_b.cu -o gpu_main_b -I./..
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c.cu -o gpu_main_c -I./..
g++ testmaker.cpp -o testmaker -std=c++0x
