
rm result_c11.out
rm result_c12.out

function tess {
	echo "Testing $1"
	printf "\n========Testing $1=========\n\n" >> result_c11.out
	./gpu_main_c11 < autotest_$1.in > autotest_$1_c11_gpu.out 2>> result_c11.out
	printf "\n========Testing $1=========\n\n" >> result_c12.out
	./gpu_main_c12 < autotest_$1.in > autotest_$1_c12_gpu.out 2>> result_c12.out
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
