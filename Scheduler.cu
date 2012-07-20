#include <stdio.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "SuperKernel.cu"
#include "Queues/QueueJobs.cu"
#include "Queues/QueueResults.cu"
#include "IncomingJobsManager.c"
#include "ResultsManager.c"


////////////////////////////////////////////////////////////////////
// The Main
////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
//Define constants
  int warp_size = 32;

  int warps = 1;   //possible input arguements
  int blocks = 1;
  
  dim3 threads(warp_size*warps/blocks, 1);
  dim3 grid(blocks, 1);

//Allocate streams, Queues
  /* Notes on these structures:
       Streams: one stream for each direction of data movement and one
                    for the Super Kernel.

       QueueJobs: The Scheduler will Enqueue jobDescriptions
                  The Super Kernel will Dequeue and execute them

       QueueResults: The Super Kernel will Enqueue jobResults
                     The Scheduler will Dequeue them and send results to its caller
  */
  cudaStream_t stream_kernel, stream_dataIn, stream_dataOut;
  cudaStreamCreate(&stream_kernel);
  cudaStreamCreate(&stream_dataIn);
  cudaStreamCreate(&stream_dataOut);

  QueueJobs    d_newJobs;
  createQueueJobs(d_newJobs, stream_dataIn);

  QueueResults d_finishedJobs;
  createQueueResults(d_finishedJobs, stream_dataOut);

//Launch the super kernel
  superKernel<<< grid, threads, 0, stream_kernel>>>(d_newJobs, d_finishedJobs);

//Launch a host thread to manage incoming jobs
  pthread_t IncomingJobManager = start_IncomingJobManager(d_newJobs);

//Launch a host thread to manage results from jobs
  pthread_t ResultsManager = start_ResultsManager(d_finishedJobs);

//wait for the managers to finish (they should never finish)
  pthread_join(IncomingJobManager);
  pthread_join(ResultsManager);

  return 0;    
}
