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


void *moveToCuda(void *val, int size){
  void *ret;
  cudaMalloc(&ret, size);
  cudaMemcpyAsync(ret, val, size, cudaMemcpyHostToDevice, stream_dataIn);
  cudaStreamSynchronize(stream_dataIn);
  return ret;
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
  int HC_numThreads = 32;
  int HC_jobs = NUMBER_OF_JOBS;

  int size = sizeof(struct JobDescription);

  printf("Starting IncomingJobs Manager\n");

  int i;
  for(i=0;i<HC_jobs;i++){
    HC_JobID = i;
    HC_JobType = (HC_JobType+1)%2;
    // launch queue jobs
    // malloc the host structure
    JobDescription *h_JobDescription = (JobDescription *) malloc(size);

    // set the values to the host structure
    h_JobDescription->JobType = HC_JobType;
    h_JobDescription->JobID = HC_JobID;
    h_JobDescription->params = moveToCuda(&SLEEP_TIME, sizeof(int));
    h_JobDescription->numThreads = HC_numThreads;

    // enqueue jobs
//    printf("Starting EnqueueJob # %d\n", HC_JobID);
    EnqueueJob(h_JobDescription, d_newJobs);
//    printf("Finished EnqueueJob # %d\n", HC_JobID);

    // free the local memory
    free(h_JobDescription);
  }
  printf("Finished Incoming Jobs Manager\n");
  return 0;
}
