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

__device__ void d_WaitIsEmpty(volatile Queue Q) {
  volatile int *s = &(Q->Size);
  int count =0;
  while(*s == 0)count++;
  return;
}

__device__ int d_IsEmpty(volatile Queue Q) {
  volatile int *s = &(Q->Size);
  return *s == 0;
}

__device__ int d_IsFull(volatile Queue Q) {
  volatile int *s = &(Q->Size);
  return *s == Q->Capacity;
}


//////////////////////////////////////////////////////////
// Host Helper Functions
//////////////////////////////////////////////////////////
int h_IsEmpty(Queue Q) {
  return Q->Size == 0;
}

int h_IsFull(Queue Q) {
  return Q->Size == Q->Capacity;
}

void printAnyErrors()
{
    cudaError e = cudaGetLastError();
    if(e!=cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString( e ) );
    }
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

/*
  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaMemcpy(h_Q, Q, sizeof(struct QueueRecord), cudaMemcpyDeviceToHost);

  printf("  Capacity, %d\n", h_Q->Capacity);
  printf("  Rear,     %d\n", h_Q->Rear);
  printf("  Size,     %d\n", h_Q->Size);
  printf("  Front,    %d\n", h_Q->Front);
*/

  return d_Q;
}

void DisposeQueue(Queue Q) {
  cudaFree(Q);
}

////////////////////////////////////////////////////////////
// Functions to modify a new jobs queue
////////////////////////////////////////////////////////////

void EnqueueJob(JobDescription *X, Queue Q) {
//called by CPU

  int copySize= sizeof(struct QueueRecord);

//printf("Start of EnqueueJob\n");

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataIn);
  cudaStreamSynchronize(stream_dataIn);

  while(h_IsFull(h_Q)){
    cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataIn);
    cudaStreamSynchronize(stream_dataIn);
  }

  printf("Queue Values at Enqueue\n");
  printf("  Capacity, %d\n", h_Q->Capacity);
  printf("  Rear,     %d\n", h_Q->Rear);
  printf("  Size,     %d\n", h_Q->Size);
  printf("  Front,    %d\n\n", h_Q->Front);

  h_Q->Size++;

  h_Q->Rear = (h_Q->Rear+1)%(h_Q->Capacity);

//printf("Middle of EnqueueJob\n");

  cudaMemcpyAsync(h_Q->Array + (h_Q->Rear)*sizeof(JobDescription *), 
                  &X, 
                  sizeof(JobDescription *),
                  cudaMemcpyHostToDevice, 
                  stream_dataIn);
  cudaStreamSynchronize(stream_dataIn);


//printf("End of EnqueueJob\n");

  cudaMemcpyAsync(Q, h_Q, copySize, cudaMemcpyHostToDevice, stream_dataIn);
  cudaStreamSynchronize(stream_dataIn);
  free(h_Q);
}

__device__ JobDescription *FrontJob(volatile Queue Q) {
//called by GPU
  getLock(Q);

  int count = 0;
  while(d_IsEmpty(Q))count++; //wait for a job

  JobDescription *result = Q->Array[Q->Front];
  releaseLock(Q);
  return result;

}

__device__ void DequeueJob(volatile Queue Q) {
//called by GPU
  getLock(Q);

  int count =0;
  while(d_IsEmpty(Q))count++; //wait for a job

  Q->Front = (Q->Front+1)%(Q->Capacity);
  Q->Size--;

  releaseLock(Q);
}

__device__ JobDescription *FrontAndDequeueJob(volatile Queue Q) {
//called by GPU
  getLock(Q);

  //d_WaitIsEmpty(Q); //wait for a job

  int count = 0;
  while(d_IsEmpty(Q))count++;

  JobDescription *result = Q->Array[Q->Front];
  Q->Front = (Q->Front+1)%(Q->Capacity);
  Q->Size--;

  releaseLock(Q);

  return result;
}

__device__ void EnqueueResult(JobDescription *X, volatile Queue Q) {
//called by GPU
  getLock(Q);

  int count =0;
  while(d_IsFull(Q))count++;

  Q->Rear = (Q->Rear+1)%(Q->Capacity);
  Q->Array[Q->Rear] = X;
  Q->Size++;

  releaseLock(Q);
}

JobDescription *FrontResult(Queue Q) {
//called by CPU
  int copySize= sizeof(struct QueueRecord);

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost,stream_dataOut);
  cudaStreamSynchronize(stream_dataOut);

  while(h_IsEmpty(h_Q)){
    cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataOut);
    cudaStreamSynchronize(stream_dataOut);
  }

  return h_Q->Array[h_Q->Front];
}

void DequeueResult(Queue Q) {
//called by CPU
  int copySize= sizeof(struct QueueRecord); // -12 for 64 bit arch, -8 for others

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataOut);
  cudaStreamSynchronize(stream_dataOut);

  while(h_IsEmpty(h_Q)){
    cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataOut);
    cudaStreamSynchronize(stream_dataOut);
  }

  h_Q->Size--;
  h_Q->Front = (h_Q->Front+1)%(h_Q->Capacity);

  cudaMemcpyAsync(Q, h_Q, copySize, cudaMemcpyHostToDevice, stream_dataOut);
  cudaStreamSynchronize(stream_dataOut);
}

JobDescription *FrontAndDequeueResult(Queue Q) {
//called by CPU
  int copySize= sizeof(struct QueueRecord);

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));

  cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataOut);
  cudaStreamSynchronize(stream_dataOut);

 // printf("%d\n", h_Q->Size);

  printf("Queue Values at Dequeue\n");
  printf("  Capacity, %d\n", h_Q->Capacity);
  printf("  Rear,     %d\n", h_Q->Rear);
  printf("  Size,     %d\n", h_Q->Size);
  printf("  Front,    %d\n", h_Q->Front);

  while(h_IsEmpty(h_Q)){
    cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataOut);
              //printf("%d\n", h_Q->Size);
    cudaStreamSynchronize(stream_dataOut);
    printf("Queue Values at Dequeue\n");
    printf("  Capacity, %d\n", h_Q->Capacity);
    printf("  Rear,     %d\n", h_Q->Rear);
    printf("  Size,     %d\n", h_Q->Size);
    printf("  Front,    %d\n", h_Q->Front);
  }

  h_Q->Size--;
  JobDescription *result = (JobDescription *)h_Q->Array + (h_Q->Front)*sizeof(JobDescription *);
  h_Q->Front = (h_Q->Front+1)%(h_Q->Capacity);

  cudaMemcpyAsync(Q, h_Q, copySize, cudaMemcpyHostToDevice, stream_dataOut);

  JobDescription *h_result = (JobDescription *) malloc(sizeof(struct JobDescription));
  cudaMemcpyAsync(h_result, result, sizeof(struct JobDescription), cudaMemcpyDeviceToHost, stream_dataOut);
  cudaStreamSynchronize(stream_dataOut);

  return h_result;
}
