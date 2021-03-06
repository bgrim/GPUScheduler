#include <stdio.h>

__device__ void clock_block(int kernel_time, int clockRate)
{ 
    int finish_clock;
    int start_time;
    for(int temp=0; temp<kernel_time; temp++){
        start_time = clock();
        finish_clock = start_time + clockRate;
        bool wrapped = finish_clock < start_time;
        while( clock() < finish_clock || wrapped) wrapped = clock()>0 && wrapped;
    }
}


__global__ void superKernel(volatile int *init, int numThreads, int *result)
{ 
    // init and result are arrays of integers where result should end up
    // being the result of incrementing all elements of init.
    // They have n elements and are (n+1) long. The should wait for the
    // first element to be set to zero
    int warp_size = 32;

    int threadID = (threadIdx.x + threadIdx.y * blockDim.x)%warp_size;
    int warpID = (threadIdx.x + threadIdx.y * blockDim.x)/warp_size;
    
    //clock_block(10,706000000);

    int count = 1;
    while(init[0]==0) count++;

    if(threadID<numThreads && warpID==0) result[threadID+1] = count;

    //__syncthreads(); //this will need to be a warp wide sync using (PTX barriers)

    if(threadID==0) result[0] = count;
}



