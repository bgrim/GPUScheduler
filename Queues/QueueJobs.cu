
#include <stdlib.h>

#define MinQueueSize (5)


////////////////////////////////////////////////////////////
// Types
////////////////////////////////////////////////////////////
struct JobDescription{
  int JobType;
  int JobID;
  void* params;
  int numThreads;
};

struct QueueJobsRecord {  //order is important in this record
  int Capacity;
  int Front;
  int Rear;
  JobDescription *Array;
  int Size;
  bool ReadLock;  //FIX
};
typedef QueueJobs *QueueJobsRecord;


////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////
__device__ void getLock(QueueJobs Q)
{
  while(atomicCAS(&(Q->ReadLock), 0, 1) != 0);
}

__device__ void releaseLock(QueueJobs Q)
{
  atomicExch(&(Q->ReadLock),0);
}

__device__ int IsEmptyJob(QueueJobs Q) {
//called by GPU
  return Q->Size == 0;
}

int IsFullJob(QueueJobs Q) {
//called by CPU
  return Q->Size == Q->Capacity;
}


////////////////////////////////////////////////////////////
// Constructor and Deconsturctor
////////////////////////////////////////////////////////////

QueueJobs CreateQueueJobs(int MaxElements) {  //FIX
  QueueJobs Q;

  if (MaxElements < MinQueueSize) {
    Error("CreateQueueJobs Error: Queue size is too small.");
  }

  Q = (QueueJobs) malloc (sizeof(struct QueueJobsRecord));
  if (Q == NULL) {
    FatalError("CreateQueueJobs Error: Unable to allocate more memory.");
  }

  Q->Array = (JobDescription *) malloc( sizeof(JobDescription) * MaxElements );
  if (Q->Array == NULL) {
    FatalError("CreateQueueJobs Error: Unable to allocate more memory.");
  }

  Q->Capacity = MaxElements;
  Q->Size = 0;
  Q->Front = 1;
  Q->Rear = 0;

  return Q;
}

void DisposeQueueJobs(QueueJobs Q) {
  if (Q != NULL) {
    free(Q->Array);
    free(Q);
  }
}


////////////////////////////////////////////////////////////
// Functions to modify a queue
////////////////////////////////////////////////////////////

void EnqueueJob(JobDescription X, QueueJobs Q) {  //make this not copy the ReadLock
//called by CPU
    QueueJobs h_Q;
    cudaMemcpy(Q, h_Q, );  //FIX

    while(IsFullJob(h_Q)) cudaMemcpy(Q, h_Q, ); //FIX

    h_Q->Size++;
    h_Q->Rear = (Q->Rear+1)%(Q->Capacity);
    h_Q->Array[Q->Rear] = X;

    cudaMemcpy(Q, h_Q, ); //FIX
}

__device__ JobDescription FrontJob(QueueJobs Q) {
//called by GPU
  getLock(Q);

  while(IsEmptyJob(Q)); //wait for a job

  JobDescription result = Q->Array[Q->Front];
  releaseLock(Q);
  return result;

}

__device__ void DequeueJob(QueueJobs Q) {
//called by GPU
  getLock(Q);

  while(IsEmptyJob(Q)); //wait for a job
  Q->Size--;
  Q->Front = (Q->Front+1)%(Q->Capacity);

  releaseLock(Q);
}

__device__ JobDescription FrontAndDequeueJob(QueueJobs Q) {
//called by GPU
  getLock(Q);

  while(IsEmptyJob(Q)); //wait for a job

  JobDescription X;
  Q->Size--;
  X = Q->Array[Q->Front];
  Q->Front = (Q->Front+1)%(Q->Capacity);

  releaseLock(Q);

  return X;
}







