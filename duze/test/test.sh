#../cpu_main < autotest_01.in > autotest_01.out
#../cpu_main_b < autotest_01.in > autotest_01_b.out
#../gpu_main_b < autotest_01.in > autotest_01_b_gpu.out

rm result_c.out
rm result_c2.out

function tess {
	echo "Testing $1"
	#printf "\n========Testing $1=========\n\n" >> result_c.out
	printf "\n========Testing $1=========\n\n" >> result_c2.out
	#../gpu_main_c < autotest_$1.in > autotest_$1_c_gpu.out 2>> result_c.out
	../gpu_main_c2 < autotest_$1.in > autotest_$1_c2_gpu.out 2>> result_c2.out
	#diff autotest_$1_c_gpu.out autotest_$1_c2_gpu.out >> result_c2.out
}

tess 01
tess 02
tess 03
tess 04
tess 04a
tess 05
tess 04b
tess 05a
tess 03a
tess 05b
#tess 04c
#tess 06a
#tess 06
#tess 07
#tess 08
