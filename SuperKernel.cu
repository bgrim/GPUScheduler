#include <stdio.h>

//#include "Queues/QueueJobs.cu"
#include "Kernels/Sleep0.cu"
#include "Kernels/Sleep1.cu"
#include "Kernels/AddSleep.cu"
#include "Kernels/MatrixSquare.cu"


__device__ JobDescription executeJob(JobDescription currentJob);

__global__ void superKernel(Queue incoming, Queue results)
{ 
    // init and result are arrays of integers where result should end up
    // being the result of incrementing all elements of init.
    // They have n elements and are (n+1) long. The should wait for the
    // first element to be set to zero
    int warp_size = 32;

    int threadID = threadIdx.x % warp_size;
    int warpID = threadIdx.x / warp_size;   //added depenency on block

    __shared__ JobDescription currentJobs[32];

    int numJobs = 1000;
    int i;
    for(i=0;i<numJobs;i++)
    {
      if(threadID==0) currentJobs[warpID] = FrontAndDequeueJob(incoming);

      JobDescription retval;
      //if(threadID<(currentJob.numThreads)) retval = executeJob(currentJob);
      retval = executeJob(currentJobs[warpID]);

      if(threadID==0) EnqueueResult(retval, results);
    }
}

__device__ JobDescription executeJob(JobDescription currentJob){

  int JobType = currentJob.JobType;

  int SleepTime = 5000;
  int clockRate = 1560000;

  // large switch
  switch(JobType){
    case 0:
      sleep0(currentJob.params, clockRate);
      break;
    case 1:
      sleep1(currentJob.params, clockRate);
      break;
    case 2:
      //addSleep(currentJob.params);
      addSleep(&SleepTime);
      break;
    case 3:
      matrixSquare(currentJob.params);
      break;
  }
  return currentJob;
}

