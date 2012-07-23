#include <stdlib.h>
#include "QueueHelpers.cu"

////////////////////////////////////////////////////////////
// Functions to modify a new jobs queue
////////////////////////////////////////////////////////////

void EnqueueJob(JobDescription X, Queue Q) {
//called by CPU
  int copySize= sizeof(struct QueueJobsRecord)-12;

  Queue h_Q = (Queue) malloc(sizeof(struct QueueJobsRecord));
  cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  while(h_IsFull(h_Q)) cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  h_Q->Size++;
  h_Q->Rear = (Q->Rear+1)%(Q->Capacity);
  h_Q->Array[Q->Rear] = X;

  cudaMemcpy(Q, h_Q, copySize, cudaMemcpyHostToDevice);
  free(h_Q);
}

__device__ JobDescription FrontJob(Queue Q) {
//called by GPU
  getLock(Q);

  while(d_IsEmpty(Q)); //wait for a job

  JobDescription result = Q->Array[Q->Front];
  releaseLock(Q);
  return result;

}

__device__ void DequeueJob(Queue Q) {
//called by GPU
  getLock(Q);

  while(d_IsEmpty(Q)); //wait for a job
  Q->Size--;
  Q->Front = (Q->Front+1)%(Q->Capacity);

  releaseLock(Q);
}

__device__ JobDescription FrontAndDequeueJob(Queue Q) {
//called by GPU
  getLock(Q);

  while(d_IsEmpty(Q)); //wait for a job

  JobDescription X;
  Q->Size--;
  X = Q->Array[Q->Front];
  Q->Front = (Q->Front+1)%(Q->Capacity);

  releaseLock(Q);

  return X;
}







