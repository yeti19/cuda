
gcc cpu_main2.c -o cpu_main
gcc cpu_main2_b.c -o cpu_main_b
nvcc gpu_main_b.cu -o gpu_main_b -I./..
g++ testmaker.cpp -o testmaker -std=c++11
testmaker < testmake_01.in > autotest_01.in

cpu_main.exe < autotest_01.in > autotest_01.out
cpu_main_b.exe < autotest_01.in > autotest_01_b.out
gpu_main_b.exe < autotest_01.in > autotest_01_b_gpu.out