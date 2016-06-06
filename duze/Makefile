all: cpu_main cpu_main_b tests

cpu-main: cpu_main2.c
	gcc cpu_main2.c -o cpu_main

cpu_main_b: cpu_main2_b.c
	gcc cpu_main2_b.c -o cpu_main_b

testmaker: testmaker.cpp
	g++ testmaker.cpp -o testmaker -std=c++11

tests: testmaker
	testmaker < testmake_01.in > autotest_01.in