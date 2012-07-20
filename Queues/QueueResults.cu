//This will be similar to QueueJobs.cu

#include <stdlib.h>

#define MinQueueSize (5)

struct JobDescription{
  int JobType;
  int JobID;
  void* params;
  int numThreads;
};

typedef QueueResults *QueueResultsRecord;

struct QueueResultsRecord {
  int Capacity;
  int Front;
  int Rear;
  int Size;
  JobDescription *Array;
};

int IsEmptyResults(QueueResults Q) {
  return Q->Size == 0;
}

int IsFullResults(QueueResults Q) {
  return Q->Size == Q->Capacity;
}

QueueResults CreateQueueResults(int MaxElements) {
  QueueResults Q;

  if (MaxElements < MinQueueSize) {
    Error("CreateQueue Error: Queue size is too small.");
  }

  Q = (QueueResults) malloc (sizeof(struct QueueResultsRecord));
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

void EnqueueResults(JobDescription X, QueueResults Q) {

  if (IsFull(Q)) {
    Error("Enqueue Error: The results queue is full.");
  } else {
    Q->Size++;
    Q->Rear = Succ(Q->Rear, Q);
    Q->Array[Q->Rear] = X;
  }

}
