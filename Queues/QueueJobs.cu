
#include <stdlib.h>

#define MinQueueSize (5)

struct JobDescription{
  int JobType;
  int JobID;
  void* params;
  int numThreads;
};

struct QueueJobsRecord {
  int Capacity;
  int Front;
  int Rear;
  int Size;
  JobDescription *Array;
};

typedef QueueJobs *QueueJobsRecord;



__device__ int IsEmptyJob(QueueJobs Q) {
//called by GPU
  return Q->Size == 0;
}

int IsFullJob(QueueJobs Q) {
//called by CPU
  return Q->Size == Q->Capacity;
}

QueueJobs CreateQueueJobs(int MaxElements) {
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


void EnqueueJob(JobDescription X, QueueJobs Q) {
//called by CPU
    QueueJobs h_Q;

    while(IsFullJob(Q, h_Q));  //wait for queue to be non-full

    Q->Size++;
    Q->Rear = (Q->Rear+1)%(Q->Capacity);
    Q->Array[Q->Rear] = X;
}

__device__ JobDescription FrontJob(QueueJobs Q) {
//called by GPU
  if (!IsEmptyJob(Q)) {
    return Q->Array[Q->Front];
  }
  Error("Front Error: The queue is empty.");

  /* Return value to avoid warnings from the compiler */
  void *r;
  return r;
}

__device__ void DequeueJob(QueueJobs Q) {
//called by GPU
  if (IsEmptyJob(Q)) {
    Error("Dequeue Error: The queue is empty.");
  } else {
    Q->Size--;
    Q->Front = (Q->Front+1)%(Q->Capacity);
  }

}

__device__ JobDescription FrontAndDequeueJob(QueueJobs Q) {
//called by GPU
  JobDescription X;

  if (IsEmptyJob(Q)) {
    Error("FrontAndDequeue Error: The queue is empty.");
  } else {
    Q->Size--;
    X = Q->Array[Q->Front];
    Q->Front = (Q->Front+1)%(Q->Capacity);
  }
  return X;

}
