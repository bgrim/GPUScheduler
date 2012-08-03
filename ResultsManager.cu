#include <stdio.h>
#include <cuda_runtime.h>
#include <pthread.h>
//#include "Queues/QueueJobs.cu"

void *main_ResultsManager(void *params);

pthread_t start_ResultsManager(Queue CompletedJobDescriptions)
{
//This should do any initializing that the results manager will
//  need and then launch a thread running main_ResultsManager,
//  returning this thread

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  pthread_t thread2;
  pthread_create( &thread2, NULL, main_ResultsManager, (void*) CompletedJobDescriptions);

  return thread2;
}


void *main_ResultsManager(void *params)
{
//The thread should read results from the queue in param and
//  print something about them to the screen.
//    --eventually this should return the result to the application
//      that requested the work.
  printf("Starting ResultsManager\n"); 

  int HC_jobs = NUMBER_OF_JOBS;
  int i;
  JobDescription currentJob;
  Queue results = (Queue)params;
  
  for(i=0;i<HC_jobs;i++){
    // front and dequeue results
//    printf("Starting to dequeue\n");
    currentJob = FrontAndDequeueResult(results);

    //JobDescription *h_matTest = malloc(sizeof(struct JobDescription));

    //cudaMemcpy(h_matTest, &currentJob, (sizeof(struct JobDescription)), cudaMemcpyDeviceToHost);


    //printf("point is to %p\n", (int *) currentJob.params);
    float *printArray = (float *)malloc((2048 * sizeof(float)));

    cudaSafeMemcpy(printArray, currentJob.params, (2048 * sizeof(float)),
                   cudaMemcpyDeviceToHost, stream_dataOut,
                   "Copying matrix back to host");

    //int i;
    //for(i = 0; i < 2048; i=1+i){
    //        printf("%f\n",printArray[i]);
    //}


    //printf("\nJob Finsihed:\n");
    //printf("  ID # %d\n", currentJob.JobID);
    //printf("  type %d\n", currentJob.JobType);
    //printf("  numT %d\n\n", currentJob.numThreads);

    cudaFree(&currentJob.params);
    //free(&currentJob);
  }
  return 0;
}
