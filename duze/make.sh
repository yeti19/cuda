
echo "Compiling cpu versions..."
gcc cpu_main2.c -o cpu_main -std=c99 -lm
gcc cpu_main2_b.c -o cpu_main_b -std=c99 -lm
echo "Compiling gpu versions..."
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_b.cu -o gpu_main_b -I./..
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c.cu -o gpu_main_c -I./..
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c2.cu -o gpu_main_c2 -I./..
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c3.cu -o gpu_main_c3 -I./..
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_d.cu -o gpu_main_d -I./..
echo "Compiling testmaker..."
g++ testmaker.cpp -o testmaker -std=c++0x

#./test/make.sh
#./test/test.sh
