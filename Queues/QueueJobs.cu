#include <stdlib.h>


struct JobDescription{
  int JobType;
  int JobID;
  void* params;
  int numThreads;
};

struct QueueRecord {
  JobDescription **Array;
  int Capacity;
  int Rear;
  int Size;
  int Front;
  int ReadLock;
};

typedef QueueRecord *Queue;


////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////
__device__ void getLock(Queue Q)
{
  while(atomicCAS(&(Q->ReadLock), 0, 1) != 0);
}

__device__ void releaseLock(Queue Q)
{
  atomicExch(&(Q->ReadLock),0);
}

__device__ int d_IsEmpty(Queue Q) {
  return Q->Size == 0;
}

__device__ int d_IsFull(Queue Q) {
  return Q->Size == Q->Capacity;
}

int h_IsEmpty(Queue Q) {
  return Q->Size == 0;
}

int h_IsFull(Queue Q) {
  return Q->Size == Q->Capacity;
}

////////////////////////////////////////////////////////////
// Constructor and Deconsturctor
////////////////////////////////////////////////////////////

Queue CreateQueue(int MaxElements) {
  Queue Q;

  //if (MaxElements < MinQueueSize) {
    //FatalError("CreateQueueJobs Error: Queue size is too small.");
  //}

  Q = (Queue) malloc (sizeof(struct QueueRecord));
  //if (Q == NULL) {
  //FatalError("CreateQueueJobs Error: Unable to allocate more memory.");
  //}

  cudaMalloc((void **)&(Q->Array), sizeof(JobDescription*)*MaxElements);

  Q->Capacity = MaxElements;
  Q->Size = 0;
  Q->Front = 1;
  Q->Rear = 0;
  Q->ReadLock = 0;

  Queue d_Q;
  cudaMalloc(&d_Q, sizeof(struct QueueRecord));
  cudaMemcpy(d_Q, Q, sizeof(struct QueueRecord), cudaMemcpyHostToDevice);
  free(Q);
  return d_Q;
}

void DisposeQueue(Queue Q) {
  if (Q != NULL) {
    cudaFree(Q->Array);
    cudaFree(Q);
  }
}

////////////////////////////////////////////////////////////
// Functions to modify a new jobs queue
////////////////////////////////////////////////////////////

void EnqueueJob(JobDescription *X, Queue Q) {
//called by CPU
  printf("Start of EnqueueJob\n");

  int copySize= sizeof(struct QueueRecord)-12;

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  printf("Middle of EnqueueJob\n");

  while(h_IsFull(h_Q)) cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  h_Q->Size++;
  h_Q->Rear = (Q->Rear+1)%(Q->Capacity);
  h_Q->Array[Q->Rear] = X;

  printf("End of EnqueueJob\n");

  cudaMemcpy(Q, h_Q, copySize, cudaMemcpyHostToDevice);
  free(h_Q);
}

__device__ JobDescription *FrontJob(Queue Q) {
//called by GPU
  getLock(Q);

  while(d_IsEmpty(Q)); //wait for a job

  JobDescription *result = Q->Array[Q->Front];
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

__device__ JobDescription *FrontAndDequeueJob(Queue Q) {
//called by GPU
  getLock(Q);

  while(d_IsEmpty(Q)); //wait for a job

  Q->Size--;
  JobDescription *result = Q->Array[Q->Front];
  Q->Front = (Q->Front+1)%(Q->Capacity);

  releaseLock(Q);

  return result;
}

__device__ void EnqueueResult(JobDescription *X, Queue Q) {
//called by GPU
  getLock(Q);

  while(d_IsFull(Q));

  Q->Size++;
  Q->Rear = (Q->Rear+1)%(Q->Capacity);
  Q->Array[Q->Rear] = X;

  releaseLock(Q);
}

JobDescription *FrontResult(Queue Q) {
//called by CPU
  int copySize= sizeof(struct QueueRecord)-12;

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  while(h_IsEmpty(h_Q)) cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  return h_Q->Array[h_Q->Front];
}

void DequeueResult(Queue Q) {
//called by CPU
  int copySize= sizeof(struct QueueRecord)-12; // -12 for 64 bit arch, -8 for others

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  while(h_IsEmpty(h_Q)) cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  h_Q->Size--;
  h_Q->Front = (h_Q->Front+1)%(h_Q->Capacity);

  cudaMemcpy(Q, h_Q, copySize, cudaMemcpyHostToDevice);
}

JobDescription *FrontAndDequeueResult(Queue Q) {
//called by CPU
  int copySize= sizeof(struct QueueRecord)-12;

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  while(h_IsEmpty(h_Q)) cudaMemcpy(h_Q, Q, copySize, cudaMemcpyDeviceToHost);

  h_Q->Size--;
  JobDescription *result = h_Q->Array[h_Q->Front];
  h_Q->Front = (h_Q->Front+1)%(h_Q->Capacity);

  cudaMemcpy(Q, h_Q, copySize, cudaMemcpyHostToDevice);

  return result;
}
