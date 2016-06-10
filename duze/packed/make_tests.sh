echo "Making tests..."

echo "Making test 01, 02, 03, 04"
testmaker < testmake_01.in > autotest_01.in
testmaker < testmake_02.in > autotest_02.in
testmaker < testmake_03.in > autotest_03.in
testmaker < testmake_03a.in > autotest_03a.in
testmaker < testmake_04.in > autotest_04.in
echo "Making test 04a"
testmaker < testmake_04a.in > autotest_04a.in
echo "Making test 04b"
testmaker < testmake_04b.in > autotest_04b.in
echo "Making test 05"
testmaker < testmake_05.in > autotest_05.in
echo "Making test 05a"
testmaker < testmake_05a.in > autotest_05a.in
echo "Making test 05b"
testmaker < testmake_05b.in > autotest_05b.in
