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
  
}

void DequeueResult(Queue Q) {
//called by CPU
  
}

JobDescription FrontAndDequeueResult(Queue Q) {
//called by CPU
  
}







