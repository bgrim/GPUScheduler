#include <stdlib.h>

struct JobDescription{
  int JobID;
  int JobType;
  int numThreads;
  void *params;
};

struct QueueRecord {
  JobDescription *Array; //Order matters here, we should improve this later
  int Capacity;          // by having two different Queues with different Orders
  int Rear;
  int Front;
  int ReadLock;
};

typedef QueueRecord *Queue;


////////////////////////////////////////////////////////////
// Locking Functions used to Sync warps access to Queues
////////////////////////////////////////////////////////////
__device__ void getLock(volatile Queue Q)
{
  while(atomicCAS(&(Q->ReadLock), 0, 1) != 0);
}

__device__ void releaseLock(volatile Queue Q)
{
  atomicExch(&(Q->ReadLock),0);
}

///////////////////////////////////////////////////////////
// Device Helper Functions
///////////////////////////////////////////////////////////

__device__ int d_IsEmpty(Queue Q) {
  volatile int *rear = &(Q->Rear);
  return (*rear+1)%Q->Capacity == Q->Front;
}

__device__ int d_IsFull(Queue Q) {
  volatile int *front = &(Q->Front);
  return (Q->Rear+2)%Q->Capacity == *front;
}


//////////////////////////////////////////////////////////
// Host Helper Functions
//////////////////////////////////////////////////////////
int h_IsEmpty(Queue Q) {
  return (Q->Rear+1)%Q->Capacity == Q->Front;
}

int h_IsFull(Queue Q) {
  return (Q->Rear+2)%Q->Capacity == Q->Front;
}

void *movePointer(void *p, int n){
   char * ret = (char *) p;
   return ((void *)(ret+n));
}

void printAnyErrors()
{
  cudaError e = cudaGetLastError();
  if(e!=cudaSuccess){
    printf("CUDA Error: %s\n", cudaGetErrorString( e ) );
  }
}



