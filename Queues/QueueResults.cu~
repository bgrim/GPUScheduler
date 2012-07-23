#include <stdlib.h>
#include "QueueHelpers.cu"

////////////////////////////////////////////////////////////
// Functions to modify a job results queue
////////////////////////////////////////////////////////////

__device__ void EnqueueResult(JobDescription X, Queue Q) {
//called by GPU
  getLock(Q);

  while(d_IsFull(Q));

  h_Q->Size++;
  h_Q->Rear = (Q->Rear+1)%(Q->Capacity);
  h_Q->Array[Q->Rear] = X;

  releaseLock(Q);
}

JobDescription FrontResult(Queue Q) {
//called by CPU
  int copySize= sizeof(struct QueueJobsRecord)-12;

  Queue h_Q = (Queue) malloc(sizeof(struct QueueJobsRecord));
  cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  while(h_Empty(h_Q)) cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  return h_Q->Array[h_Q->Front];
}

void DequeueResult(Queue Q) {
//called by CPU
  int copySize= sizeof(struct QueueJobsRecord)-12;

  Queue h_Q = (Queue) malloc(sizeof(struct QueueJobsRecord));
  cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  while(h_Empty(h_Q)) cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  h_Q->Size--;
  h_Q->Front = (h_Q->Front+1)%(h_Q->Capacity);

  cudaMemcpy(Q, h_Q, copySize, cudaMemcpyHostToDevice);
}

JobDescription FrontAndDequeueResult(Queue Q) {
//called by CPU
  int copySize= sizeof(struct QueueJobsRecord)-12;

  Queue h_Q = (Queue) malloc(sizeof(struct QueueJobsRecord));
  cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  while(h_Empty(h_Q)) cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  JobDescription result;
  h_Q->Size--;
  result = h_Q->Array[h_Q->Front];
  h_Q->Front = (h_Q->Front+1)%(h_Q->Capacity);

  cudaMemcpy(Q, h_Q, copySize, cudaMemcpyHostToDevice);

  return result;
}







