#include <stdio.h>

//#include "Queues/QueueJobs.cu"
#include "Kernels/Sleep0.cu"

__device__ JobDescription executeJob(JobDescription currentJob);

__global__ void superKernel(volatile Queue incoming, Queue results)
{ 
    // init and result are arrays of integers where result should end up
    // being the result of incrementing all elements of init.
    // They have n elements and are (n+1) long. The should wait for the
    // first element to be set to zero
    int warp_size = 32;

    int threadID = threadIdx.x % warp_size;
    //int warpID = threadIdx.x / warp_size;   //added depenency on block

    int numJobs = 1;
    int i;

    for(i=0;i<numJobs;i++)
    {
      JobDescription currentJob;

      if(threadID==0) currentJob = FrontAndDequeueJob(incoming);

      JobDescription retval = currentJob;
      //if(threadID<(currentJob->numThreads)) retval = executeJob(currentJob);

      if(threadID==0) EnqueueResult(retval, results);
    }
}

__device__ JobDescription executeJob(JobDescription currentJob){

  int JobType = currentJob->JobType;

  int SleepTime = 1;
  int clockRate = 1; //706000000;

  // large switch
  switch(JobType){
  case 0:
    // call case 0
    sleep0(SleepTime, clockRate);
  }
  return currentJob;
}

