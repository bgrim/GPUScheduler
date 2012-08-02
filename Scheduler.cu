#include <stdio.h>
#include <cuda_runtime.h>
#include <pthread.h>

cudaStream_t stream_dataIn, stream_dataOut; //this is probably not the best way to make these
                                            // streams globally known, oh well.
int SLEEP_TIME;
int NUMBER_OF_JOBS;

#include "Queues/QueueJobs.cu"
#include "IncomingJobsManager.cu"
#include "ResultsManager.cu"
#include "SuperKernel.cu"

////////////////////////////////////////////////////////////////////
// The Main
////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
//Define constants
  int warp_size = 32;

  int warps = 32;   //possible input arguements
  int blocks = 7;
  NUMBER_OF_JOBS = 224;
  SLEEP_TIME = 1000;
  if(argc>4){
    warps = atoi(argv[1]);
    blocks = atoi(argv[2]);
    NUMBER_OF_JOBS = atoi(argv[3]);
    SLEEP_TIME = atoi(argv[4]);
  }
  
  dim3 threads(warp_size*warps, 1);
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
  cudaStream_t stream_kernel;
  cudaStreamCreate(&stream_kernel);
  cudaStreamCreate(&stream_dataIn);
  cudaStreamCreate(&stream_dataOut);

  Queue d_newJobs = CreateQueue(25600); //FIX, make this use stream_dataIn

  Queue d_finishedJobs = CreateQueue(25600); //FIX, make this use stream_dataOut


//Launch the super kernel
  superKernel<<< grid, threads, 0, stream_kernel>>>(d_newJobs, d_finishedJobs);

//Launch a host thread to manage incoming jobs
  pthread_t IncomingJobManager = start_IncomingJobsManager(d_newJobs);

//Launch a host thread to manage results from jobs
  pthread_t ResultsManager = start_ResultsManager(d_finishedJobs);


//wait for the managers to finish (they should never finish)
  void * r;
  pthread_join(IncomingJobManager, &r);
  pthread_join(ResultsManager, &r);

  printf("Both managers have finished\n");
  printf("Destroying Streams...\n");
  cudaStreamDestroy(stream_kernel);
  cudaStreamDestroy(stream_dataIn);
  cudaStreamDestroy(stream_dataOut);

  printf("Destorying Queues...\n");

  DisposeQueue(d_newJobs);

  DisposeQueue(d_finishedJobs);

  printf("Exiting Main\n\n");

  return 0;    
}
