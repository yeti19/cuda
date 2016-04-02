#/bin/bash

for i in {1..8}
do
	echo -n "Test $i.. " >> testing.log
	$1 < $2/test${i}.in > gout
	cmp $2/skrypt/cpu-solved/cout${i} gout >> testing.log
	if [ $? -eq 0 ]
	then
		echo "OK" >> testing.log
	else
		echo "ERR" >> testing.log
	fi
done 
rm gout
