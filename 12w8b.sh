#! /bin/bash 

#Sleep Time is currently not used            
sleepTime=10


# do the overall test, this many times
for i in {1..9}
do
    warps=1
    blocks=1
    numJobs=2

    for k in {1..9}
    do

	echo $sleepTime
	echo $numJobs
	(/usr/bin/time -f "%e" ./bin/run $warps $blocks $numJobs $sleepTime) 2>> logs/12w8b.txt
	
	#warps=$(($warps+$warps))
        numJobs=$(($numJobs+$numJobs))
    done
    sleepTime=$(($sleepTime+$sleepTime))
done
