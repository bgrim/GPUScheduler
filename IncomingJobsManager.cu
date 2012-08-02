#include <stdio.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <sys/time.h>

struct timeval tp;


double getTime_usec() {
    gettimeofday(&tp, NULL);
    return static_cast<double>(tp.tv_sec) * 1E6
            + static_cast<double>(tp.tv_usec);
}


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
  cudaSafeMemcpy(ret, val, size, 
                 cudaMemcpyHostToDevice, stream_dataIn, 
                 "in moveToCuda of IncomingJobsManager.cu");
  return ret;
}

float *makeMatrix(){
  int ROW = 32;
  int COLUMN = 32;

  int a=0, b=0;

  float *stuff = (float *) malloc(2*(COLUMN * ROW * sizeof(float)));
  for(a=0; a<ROW;a++)
    {
      for(b=0; b<COLUMN;b++)
        {
	  stuff[a + b * ROW]=((float)rand())/((float) RAND_MAX);
	  stuff[a + b * ROW + ROW * COLUMN] = 0.0;
	}
    }
  //int i;
  //for(i = 0; i <2048; i=i+32){
  //  printf("%f\n",stuff[i]);
  //}
  return stuff;
}


void *main_IncomingJobsManager(void *p)
{
//The thread should get job descriptions some how and Enqueue them
//  into the queue in params
//    --eventually this should get jobs from an external application
//      but will probably just be hardcoded at first or a parameter

  Queue d_newJobs = (Queue) p;

  // Hard code for testing
  int HC_JobType = 3; // hard code the job type for sleeps
  int HC_JobID;
  int HC_numThreads = 32;
  int HC_jobs = NUMBER_OF_JOBS;
  int HC_matrixWidth = 32;
  int HC_matrixSize = HC_matrixWidth * HC_matrixWidth;

  int size = sizeof(struct JobDescription);

  printf("Starting IncomingJobs Manager\n");

  int i;
  for(i=0;i<HC_jobs;i++){
    HC_JobID = i;
//    HC_JobType = (HC_JobType+1)%2;
    // launch queue jobs
    // malloc the host structure
    JobDescription *h_JobDescription = (JobDescription *) malloc(size);

    // set the values to the host structure
    h_JobDescription->JobType = HC_JobType;
    h_JobDescription->JobID = HC_JobID;
    h_JobDescription->params = moveToCuda(makeMatrix(), (2 * sizeof(float) * HC_matrixSize)); // working on float matrix
    h_JobDescription->numThreads = HC_numThreads;

    //printf("Original pointer to %p\n", (int *) h_JobDescription->params);

    // enqueue jobs
    EnqueueJob(h_JobDescription, d_newJobs);
    printf("Finished EnqueueJob # %d\n", HC_JobID);

    // free the local memory
    free(h_JobDescription);
  }
  printf("Finished Incoming Jobs Manager\n");
  return 0;
}
