
#echo "Compiling cpu versions..."
#gcc cpu_main2.c -o cpu_main -std=c99 -lm
#gcc cpu_main2_b.c -o cpu_main_b -std=c99 -lm
echo "Compiling gpu versions..."
#nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_b.cu -o gpu_main_b -I./..
#nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c.cu -o gpu_main_c -I./..
#nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c2.cu -o gpu_main_c2 -I./..
#nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c3.cu -o gpu_main_c3 -I./..
#nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c4.cu -o gpu_main_c4 -I./..
#nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c5.cu -o gpu_main_c5 -I./..
#nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c6.cu -o gpu_main_c6 -I./..
#nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c7.cu -o gpu_main_c7 -I./..
#nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c8.cu -o gpu_main_c8 -I./..
#nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c9.cu -o gpu_main_c9 -I./..
#nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c10.cu -o gpu_main_c10 -I./..
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c11.cu -o gpu_main_c11 -I./..
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_c12.cu -o gpu_main_c12 -I./..
#nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_d.cu -o gpu_main_d -I./..
nvcc -arch=sm_37 --ptxas-options=-v -use_fast_math gpu_main_d2.cu -o gpu_main_d2 -I./..
echo "Compiling testmaker..."
g++ testmaker.cpp -o testmaker -std=c++0x

#./test/make.sh
#./test/test.sh
