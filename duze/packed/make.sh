echo "Compiling cpu versions..."
gcc cpu_main2_b.c -o cpu_b -std=c99 -lm
echo "Compiling gpu versions..."
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c11.cu -o gpu_c11
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c12.cu -o gpu_c12
echo "Compiling testmaker..."
g++ testmaker.cpp -o testmaker -std=c++0x
