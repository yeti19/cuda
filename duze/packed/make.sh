
echo "Compiling gpu versions..."
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c11.cu -o gpu_main_c11
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c12.cu -o gpu_main_c12
echo "Compiling testmaker..."
g++ testmaker.cpp -o testmaker -std=c++0x
