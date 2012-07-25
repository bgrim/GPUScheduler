run:
	nvcc -g -G -arch=sm_11 Scheduler.cu -o run
clean:
	rm run