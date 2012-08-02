run:
	nvcc -g -G -arch=sm_11 Scheduler.cu -o bin/run
clean:
	rm bin/run