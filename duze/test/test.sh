../cpu_main < autotest_01.in > autotest_01.out
../cpu_main_b < autotest_01.in > autotest_01_b.out
../gpu_main_b < autotest_01.in > autotest_01_b_gpu.out

rm result_c.out

../gpu_main_c < autotest_01.in > autotest_01_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_02.in > autotest_02_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_03.in > autotest_03_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_03a.in > autotest_03a_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_04.in > autotest_04_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_04a.in > autotest_04a_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_04b.in > autotest_04b_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_04c.in > autotest_04c_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_05.in > autotest_05_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_05a.in > autotest_05a_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_05b.in > autotest_05b_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_06.in > autotest_06_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_06a.in > autotest_06a_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_07.in > autotest_07_c_gpu.out 2>> result_c.out
../gpu_main_c < autotest_08.in > autotest_08_c_gpu.out 2>> result_c.out
