#include <stdlib.h>
#include "QueueHelpers.cu"

////////////////////////////////////////////////////////////
// Constructor and Deconsturctor
////////////////////////////////////////////////////////////

Queue CreateQueue(int MaxElements) {
  Queue Q = (Queue) malloc (sizeof(struct QueueRecord));

  cudaMalloc((void **)&(Q->Array), sizeof(JobDescription)*MaxElements);

  Q->Capacity = MaxElements;
  Q->Front = 1;
  Q->Rear = 0;
  Q->ReadLock = 0;

  Queue d_Q;
  cudaMalloc(&d_Q, sizeof(struct QueueRecord));
  cudaSafeMemcpy(d_Q, Q, sizeof(struct QueueRecord), 
                 cudaMemcpyHostToDevice, stream_dataIn, 
                 "Copying initial queue to device");
  free(Q);
  return d_Q;
}

void DisposeQueue(Queue d_Q) {
  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaSafeMemcpy(h_Q, d_Q, sizeof(struct QueueRecord), 
                 cudaMemcpyDeviceToHost, stream_dataIn,
                 "DisposeQueue, Copying Queue to get array pointer");
  cudaFree(h_Q->Array);
  free(h_Q);
  cudaFree(d_Q);
}

////////////////////////////////////////////////////////////
// Host Functions to Change Queues
////////////////////////////////////////////////////////////

void EnqueueJob(JobDescription *h_JobDescription, Queue Q) {
//called by CPU

  int copySize= sizeof(struct QueueRecord);

  //printf("Start of EnqueueJob\n");

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaSafeMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataIn,
                 "EnqueueJob, Getting Queue");
/*
  printf("Queue Values at Enqueue\n");
  printf("  Capacity, %d\n", h_Q->Capacity);
  printf("  Rear,     %d\n", h_Q->Rear);
  printf("  Front,    %d\n\n", h_Q->Front);
*/
  while(h_IsFull(h_Q)){
    pthread_yield();
    cudaSafeMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataIn,
                    "EnqueueJob, Getting Queue again...");
  }

  // floating point exception from mod capacity if 0 or -n
  h_Q->Rear = (h_Q->Rear+1)%(h_Q->Capacity);

  // set job description
  cudaSafeMemcpy( h_Q->Array + h_Q->Rear,
                  h_JobDescription, 
                  sizeof(JobDescription),
                  cudaMemcpyHostToDevice, 
                  stream_dataIn,
                  "EnqueueJob, Writing Job Description");

  cudaSafeMemcpy(movePointer(Q, 12), movePointer(h_Q, 12), 
		 sizeof(int), cudaMemcpyHostToDevice, stream_dataIn,
                 "EnqueueJob, Updating Queue");

  free(h_Q);
}

JobDescription FrontAndDequeueResult(Queue Q) {
//called by CPU

  int copySize= sizeof(struct QueueRecord);

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));

  cudaSafeMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataOut,
                 "FandDJob, Getting Queue");
/*
  printf("Queue Values at Dequeue\n");
  printf("  Capacity, %d\n", h_Q->Capacity);
  printf("  Rear,     %d\n", h_Q->Rear);
  printf("  Front,    %d\n", h_Q->Front);
*/
  while(h_IsEmpty(h_Q)){
    pthread_yield();
    cudaSafeMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataOut,
                   "FandDJob, Getting Queue again...");
  }

  JobDescription *result = (JobDescription *) malloc(sizeof(JobDescription));

  cudaSafeMemcpy(result, &h_Q->Array[h_Q->Front], sizeof(JobDescription), cudaMemcpyDeviceToHost, stream_dataOut,
                 "FandDJob, Getting Job Description");

  h_Q->Front = (h_Q->Front+1)%(h_Q->Capacity);

  cudaSafeMemcpy( movePointer(Q, 16), movePointer(h_Q, 16), 
		  sizeof(int), cudaMemcpyHostToDevice, stream_dataOut,
                  "FandDJob, Updating Queue");

  cudaFree(&h_Q->Array[h_Q->Front]);

  free(h_Q);

  return *result;
}


////////////////////////////////////////////////////////////
// Device Functions to Change Queues
////////////////////////////////////////////////////////////

__device__ JobDescription FrontAndDequeueJob(Queue Q) {
//called by GPU
  getLock(Q);
 
  int count = 0;
  while(d_IsEmpty(Q))count++;

  JobDescription result = Q->Array[Q->Front];
  volatile int *front = &Q->Front;
  *front = ((*front)+1)%(Q->Capacity);

  releaseLock(Q);

  return result;
}

__device__ void EnqueueResult(JobDescription X, Queue Q) {
//called by GPU
  getLock(Q);

  int count =0;
  while(d_IsFull(Q))count++;
  int temp = (Q->Rear+1)%(Q->Capacity); 
  Q->Array[temp] = X;
  volatile int *rear = &Q->Rear;
  *rear = temp;

  releaseLock(Q);
}




