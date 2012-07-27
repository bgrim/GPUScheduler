#include <stdio.h>
#include <cuda_runtime.h>
#include <pthread.h>

//#include "Queues/QueueJobs.cu"
void *main_IncomingJobsManager(void *p);

pthread_t start_IncomingJobsManager(Queue d_newJobs)
{
//This should do any initializing that the incoming jobs
//  manager will need and then launch a thread running
//  main_IncomingJobsManager(  ), returning that thread

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  pthread_t thread1;
  pthread_create( &thread1, &attr, main_IncomingJobsManager, (void*) d_newJobs);
  pthread_attr_destroy(&attr);
  return thread1;
}


void *main_IncomingJobsManager(void *p)
{
//The thread should get job descriptions some how and Enqueue them
//  into the queue in params
//    --eventually this should get jobs from an external application
//      but will probably just be hardcoded at first or a parameter

  Queue d_newJobs = (Queue) p;

  // Hard code for testing
  int HC_JobType = 0; // hard code the job type for sleeps
  int HC_JobID;
  void* HC_params = 0;
  int HC_numThreads = 1;
  int HC_jobs = 1;

  int size = sizeof(struct JobDescription);

  int i;
  for(i=0;i<HC_jobs;i++){
    HC_JobID = i;
    // launch queue jobs
    // malloc the host structure
    JobDescription *h_JobDescription = (JobDescription *) malloc(size);

    // set the values to the host structure
    h_JobDescription->JobType = HC_JobType;
    h_JobDescription->JobID = HC_JobID;
    h_JobDescription->params = HC_params;
    h_JobDescription->numThreads = HC_numThreads;

    JobDescription *d_JobDescription;

    // cuda Malloc for the structure
    cudaMalloc((void **) &d_JobDescription, size);

    // cuda mem copy
    cudaMemcpyAsync(d_JobDescription, h_JobDescription, size, cudaMemcpyHostToDevice, stream_dataIn);
    cudaStreamSynchronize(stream_dataIn);

    printf("EnqueueJob # %d\n", HC_JobID);
    // enqueue jobs
    EnqueueJob(d_JobDescription, d_newJobs);

    // free the local memory
    free(h_JobDescription);
  }
  printf("Incoming Jobs Manager is finished\n");
  return 0;
}
