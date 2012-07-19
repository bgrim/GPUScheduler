#include <stdio.h>

__global__ void superKernel(int *init, int numThreads,int *result)
{ 
    // init and result are arrays of integers where result should end up
    // being the result of incrementing all elements of init.
    // They have n elements and are (n+1) long. The should wait for the
    // first element to be set to zero
    int warp_size = 32;

    int threadID = threadIdx.x % warp_size;
    int warpID = threadIdx.x / warp_size;

    while(init[0]!=0);

    if(threadID<numThreads && warpID==0) result[threadID+1] = init[threadID+1]+1;

    __syncthreads(); //this will need to be a warp wide sync using (PTX barriers)

    if(threadID==0) result[0] = 0;
}



