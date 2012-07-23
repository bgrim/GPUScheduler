#include <stdio.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "Kernels/SuperKernel.cu"
#include "Queues/QueueJobs.cu"
#include "Queues/QueueResults.cu"


pthread_t start_ResultsManager(Queue CompletedJobDescriptions)
{
//This should do any initializing that the results manager will
//  need and then launch a thread running main_ResultsManager,
//  returning this thread

  pthread_t thread1;
  pthread_create( &thread1, NULL, main_IncomingJobsManager, (void*) CompletedJobDescriptions);
}


void *main_ResultsManager(void *params)
{
//The thread should read results from the queue in param and
//  print something about them to the screen.
//    --eventually this should return the result to the application
//      that requested the work.
 
  int HC_jobs =64;
  int i;
  JobDescription currentJob;
  results = (Queue)params;
  
  for(i=0;i<HC_jobs;i++){
    // front and dequeue results
    currentJob = FrontAndDeqeueResults(results);
    printf("Job with ID # %d finished.", currentJob->JobID);
  }
}
