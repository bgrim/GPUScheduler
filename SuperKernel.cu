#include <stdio.h>
//#include "Queues/QueueJobs.cu"
#include "Kernels/Sleep0.cu"

__device__ JobDescription *executeJob(JobDescription *currentJob);

__global__ void superKernel(Queue incoming, Queue results)
{ 
    // init and result are arrays of integers where result should end up
    // being the result of incrementing all elements of init.
    // They have n elements and are (n+1) long. The should wait for the
    // first element to be set to zero
    int warp_size = 32;

    int threadID = threadIdx.x % warp_size;
    //int warpID = threadIdx.x / warp_size;   //added depenency on block

    while(true)
    {
      JobDescription *currentJob;

      if(threadID==0) currentJob = FrontAndDequeueJob(incoming);

      __syncthreads(); //see below comment

      JobDescription *retval;
      if(threadID<(currentJob->numThreads)) retval = executeJob(currentJob);

      __syncthreads(); //this will need to be a warp wide sync using (PTX barriers)

      if(threadID==0) EnqueueResult(retval, results);
    }
}

__device__ JobDescription *executeJob(JobDescription *currentJob){

  int JobType = currentJob->JobType;

  int SleepTime = 1;
  int clockRate = 706000000;

  // large switch
  switch(JobType){
  case 0:
    // call case 0
    sleep0(SleepTime, clockRate);
  }
  return currentJob;
}

