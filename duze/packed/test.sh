rm result_b.out
rm result_c11.out
rm result_c12.out

function tesss {
    printf "\n========Testing $1=========\n\n" >> result_$2.out
    ./$2 < autotest_$1.in > autotest_$1_$2.out 2>> result_$2.out
}

function tess {
	echo "Testing $1"
    tesss $1 cpu_b
    tesss $1 gpu_c11
    tesss $1 gpu_c12
}

tess 01
tess 02
tess 03
#tess 04
#tess 04a
#tess 05
#tess 04b
#tess 05a
#tess 03a
#tess 05b
