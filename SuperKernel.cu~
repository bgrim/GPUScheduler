#include <stdio.h>
#include "Queues/QueueJobs.cu"
#include "Queues/QueueResults.cu"
#include "Kernels/Sleep.cu"
#include "Kernels/MatrixMultiply.cu"


__global__ void superKernel(QueueJobs incoming, QueueResults results)
{ 
    // init and result are arrays of integers where result should end up
    // being the result of incrementing all elements of init.
    // They have n elements and are (n+1) long. The should wait for the
    // first element to be set to zero
    int warp_size = 32;

    int threadID = threadIdx.x % warp_size;
    int warpID = threadIdx.x / warp_size;   //added depenency on block

    while(true)
    {
      JobDescription currentJob;

      if(threadID==0) currentJob = DequeueJob(incoming);

      __syncthreads(); //see below comment

      JobResults retval = executeJob(currentJob);

      __syncthreads(); //this will need to be a warp wide sync using (PTX barriers)

      if(threadID==0) EnqueueResult(retval, results);
    }
}

__device__ JobResults executeJob(JobDescription currentJob){
  // set the JobType
  int JobType = currentJob->JobType;
  
  int sleepTime = 1000;

  // large switch
  switch(JobType){
  case 0:
    // call case 0
    sleep0(sleepTime);
  }
  case 1:
    // call case 1
    sleep1(sleepTime);
  }
  case 2:
    // call case 2
    sleep2(sleepTime);
  }
}

