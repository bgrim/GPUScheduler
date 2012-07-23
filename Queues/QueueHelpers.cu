#include <stdlib.h>

#define MinQueueSize (5)

struct JobDescription{
  int JobType;
  int JobID;
  void* params;
  int numThreads;
};

struct QueueRecord {
  JobDescription *Array;
  int Capacity;
  int Rear;
  int Size;
  int Front;
  int ReadLock;
};
typedef Queue *QueueJobsRecord;


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

QueueJobs CreateQueue(int MaxElements) {
  QueueJobs Q;

  if (MaxElements < MinQueueSize) {
    Error("CreateQueueJobs Error: Queue size is too small.");
  }

  Q = (QueueJobs) malloc (sizeof(struct QueueJobsRecord));
  if (Q == NULL) {
    FatalError("CreateQueueJobs Error: Unable to allocate more memory.");
  }

  cudaMalloc(Q->Array, sizeof(JobDescription)*MaxElements);

  Q->Capacity = MaxElements;
  Q->Size = 0;
  Q->Front = 1;
  Q->Rear = 0;
  Q->ReadLock = 0;

  Queue d_Q;
  cudaMalloc(d_Q, sizeof(struct QueueJobsRecord));
  cudaMemcpy(d_Q, Q, sizeof(struct QueueJobsRecord), cudaMemcpyHostToDevice);
  free()
  return d_Q;
}

void DisposeQueue(Queue Q) {
  if (Q != NULL) {
    cudaFree(Q->Array);
    cudaFree(Q);
  }
}






