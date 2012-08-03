#! /bin/bash 

#Sleep Time is currently not used            
sleepTime=0


# do the overall test, this many times
for i in {1..50}
do
    warps=1
    blocks=1
    numJobs=$(($warps*$blocks))

    for k in {1..1}
    do
	(/usr/bin/time -f "%e" ./bin/run $warps $blocks $numJobs $sleeptime) 2>> logs/overhead.txt

	#warps=$(($warps+$warps))
        #numJobs=$(($warps*$blocks*64))
    done
  #sleepTime=$(($sleepTime+$sleepTime))
done
