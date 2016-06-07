
gcc cpu_main2.c -o cpu_main
gcc cpu_main2_b.c -o cpu_main_b
nvcc -v gpu_main_b.cu -o gpu_main_b -I./..
nvcc -v gpu_main_c.cu -o gpu_main_c -I./..
g++ testmaker.cpp -o testmaker -std=c++11
