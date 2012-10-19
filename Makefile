run:
	nvcc -g -I/usr/include -G -arch=sm_11 -lpthread Scheduler.cu -o bin/run
clean:
	rm bin/run
