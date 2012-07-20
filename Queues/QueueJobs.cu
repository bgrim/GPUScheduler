
#include <stdlib.h>

#define MinQueueSize (5)

struct JobDescription{
  int JobType;
  int JobID;
  void* params;
  int numThreads;
};

typedef QueueJobs *QueueJobsRecord;

struct QueueJobsRecord {
  int Capacity;
  int Front;
  int Rear;
  int Size;
  JobDescription *Array;
};

int IsEmpty(QueueJobs Q) {
  return Q->Size == 0;
}

int IsFull(QueueJobs Q) {
  return Q->Size == Q->Capacity;
}

QueueJobs CreateQueueJobs(int MaxElements) {
  QueueJobs Q;

  if (MaxElements < MinQueueSize) {
    Error("CreateQueue Error: Queue size is too small.");
  }

  Q = (QueueJobs) malloc (sizeof(struct QueueJobsRecord));
  if (Q == NULL) {
    FatalError("CreateQueue Error: Unable to allocate more memory.");
  }

  Q->Array = (JobDescription *) malloc( sizeof(JobDescription) * MaxElements );
  if (Q->Array == NULL) {
    FatalError("CreateQueue Error: Unable to allocate more memory.");
  }

  Q->Capacity = MaxElements;
  MakeEmpty(Q);

  return Q;
}

void MakeEmpty(QueueJobs Q) {

  Q->Size = 0;
  Q->Front = 1;
  Q->Rear = 0;

}

void DisposeQueue(QueueJobs Q) {
  if (Q != NULL) {
    free(Q->Array);
    free(Q);
  }
}

static int Succ(int Value, QueueJobs Q) {
  if (++Value == Q->Capacity) {
    Value = 0;
  }
  return Value;
}

void Enqueue(JobDescription X, QueueJobs Q) {

  if (IsFull(Q)) {
    Error("Enqueue Error: The queue is full.");
  } else {
    Q->Size++;
    Q->Rear = Succ(Q->Rear, Q);
    Q->Array[Q->Rear] = X;
  }

}

JobDescription Front(QueueJobs Q) {

  if (!IsEmpty(Q)) {
    return Q->Array[Q->Front];
  }
  Error("Front Error: The queue is empty.");

  /* Return value to avoid warnings from the compiler */
  //  return 0;
  void *r;
  return r;
}

void Dequeue(QueueJobs Q) {

  if (IsEmpty(Q)) {
    Error("Dequeue Error: The queue is empty.");
  } else {
    Q->Size--;
    Q->Front = Succ(Q->Front, Q);
  }

}

JobDescription FrontAndDequeue(QueueJobs Q) {

  //  JobDescription X = 0;
  JobDescription X;

  if (IsEmpty(Q)) {
    Error("FrontAndDequeue Error: The queue is empty.");
  } else {
    Q->Size--;
    X = Q->Array[Q->Front];
    Q->Front = Succ(Q->Front, Q);
  }
  return X;

}
