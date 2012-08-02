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
  volatile int *s = &(Q->Rear);
  return (*s+1)%Q->Capacity == Q->Front;
}

__device__ int d_IsFull(Queue Q) {
  volatile int *s = &(Q->Rear);
  return (*s+2)%Q->Capacity == Q->Front;
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

void synchronizeAndPrint(cudaStream_t stream, char *s){
  cudaError_t e = cudaStreamSynchronize(stream);
  if(e!=cudaSuccess){
    printf("CUDA Error:   %s   at %s\n", cudaGetErrorString( e ), s);
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

  cudaMalloc((void **)&(Q->Array), sizeof(JobDescription)*MaxElements);

  Q->Capacity = MaxElements;
  Q->Front = 1;
  Q->Rear = 0;
  Q->ReadLock = 0;

  Queue d_Q;
  cudaMalloc(&d_Q, sizeof(struct QueueRecord));
  cudaMemcpy(d_Q, Q, sizeof(struct QueueRecord), cudaMemcpyHostToDevice);
  free(Q);


  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaMemcpy(h_Q, Q, sizeof(struct QueueRecord), cudaMemcpyDeviceToHost);
/*
  printf("  Capacity, %d\n", h_Q->Capacity);
  printf("  Rear,     %d\n", h_Q->Rear);
  printf("  Front,    %d\n", h_Q->Front);
*/

  return d_Q;
}

void DisposeQueue(Queue Q) {
  //FIX, needs to free the array with Queue
  cudaFree(Q);
}

////////////////////////////////////////////////////////////
// Functions to modify a new jobs queue
////////////////////////////////////////////////////////////

void EnqueueJob(JobDescription *h_JobDescription, Queue Q) {
//called by CPU

  int copySize= sizeof(struct QueueRecord);

  //printf("Start of EnqueueJob\n");

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataIn);
  synchronizeAndPrint(stream_dataIn, "EnqueueJob, Getting Queue");
/*
  printf("Queue Values at Enqueue\n");
  printf("  Capacity, %d\n", h_Q->Capacity);
  printf("  Rear,     %d\n", h_Q->Rear);
  printf("  Front,    %d\n\n", h_Q->Front);
*/
  while(h_IsFull(h_Q)){
    pthread_yield();
    cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataIn);
    synchronizeAndPrint(stream_dataIn, "EnqueueJob, Getting Queue again...");
  }

  h_Q->Rear = (h_Q->Rear+1)%(h_Q->Capacity);

  //printf("Middle of EnqueueJob\n");

  cudaMemcpyAsync(h_Q->Array + h_Q->Rear,
                  h_JobDescription, 
                  sizeof(JobDescription),
                  cudaMemcpyHostToDevice, 
                  stream_dataIn);
  synchronizeAndPrint(stream_dataIn, "EnqueueJob, Writing Job Description");


  //printf("End of EnqueueJob\n");

  cudaMemcpyAsync(movePointer(Q, 12), movePointer(h_Q, 12), 
                   sizeof(int), cudaMemcpyHostToDevice, stream_dataIn);
  synchronizeAndPrint(stream_dataIn, "EnqueueJob, Updating Queue");

  free(h_Q);
}

__device__ JobDescription FrontJob(Queue Q) {
//called by GPU
  getLock(Q);

  int count = 0;
  while(d_IsEmpty(Q))count++; //wait for a job

  JobDescription result = Q->Array[Q->Front];
  releaseLock(Q);
  return result;

}

__device__ void DequeueJob(Queue Q) {
//called by GPU
  getLock(Q);

  int count =0;
  while(d_IsEmpty(Q))count++; //wait for a job

  Q->Front = (Q->Front+1)%(Q->Capacity);

  releaseLock(Q);
}

__device__ JobDescription FrontAndDequeueJob(Queue Q) {
//called by GPU
  getLock(Q);
 
  int count = 0;
  while(d_IsEmpty(Q))count++;

  JobDescription result = Q->Array[Q->Front];
  volatile int *front = &Q->Front;
  *front = ((*front)+1)%(Q->Capacity);
  //Q->Front = (Q->Front+1)%(Q->Capacity);

  releaseLock(Q);

  return result;
}

__device__ void EnqueueResult(JobDescription X, Queue Q) {
//called by GPU
  getLock(Q);

  int count =0;
  while(d_IsFull(Q))count++;
  int temp = (Q->Rear+1)%(Q->Capacity); 
  Q->Array[temp] = X;
  volatile int *rear = &Q->Rear;
  *rear = temp;

  releaseLock(Q);
}

JobDescription FrontResult(Queue Q) {
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
  int copySize= sizeof(struct QueueRecord); 

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));
  cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataOut);
  cudaStreamSynchronize(stream_dataOut);

  while(h_IsEmpty(h_Q)){
    cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataOut);
    cudaStreamSynchronize(stream_dataOut);
  }

  h_Q->Front = (h_Q->Front+1)%(h_Q->Capacity);

  cudaMemcpyAsync(movePointer(Q, 16), movePointer(h_Q, 16), 
                   sizeof(int), cudaMemcpyHostToDevice, stream_dataOut);
  cudaStreamSynchronize(stream_dataOut);
}

JobDescription FrontAndDequeueResult(Queue Q) {
//called by CPU

  int copySize= sizeof(struct QueueRecord);

  Queue h_Q = (Queue) malloc(sizeof(struct QueueRecord));

  cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataOut);
  synchronizeAndPrint(stream_dataOut, "FandDJob, Getting Queue");

/*
  printf("Queue Values at Dequeue\n");
  printf("  Capacity, %d\n", h_Q->Capacity);
  printf("  Rear,     %d\n", h_Q->Rear);
  printf("  Front,    %d\n", h_Q->Front);
*/
  while(h_IsEmpty(h_Q)){
    pthread_yield();
    cudaMemcpyAsync(h_Q, Q, copySize, cudaMemcpyDeviceToHost, stream_dataOut);
    synchronizeAndPrint(stream_dataOut, "FandDJob, Getting Queue again...");

/*
    printf("Queue Values at Dequeue\n");
    printf("  Capacity, %d\n", h_Q->Capacity);
    printf("  Rear,     %d\n", h_Q->Rear);
    printf("  Front,    %d\n", h_Q->Front);
*/
  }

  JobDescription *result = (JobDescription *) malloc(sizeof(JobDescription));

  cudaMemcpyAsync(result, &h_Q->Array[h_Q->Front], sizeof(JobDescription), cudaMemcpyDeviceToHost, stream_dataOut);
  synchronizeAndPrint(stream_dataOut, "FandDJob, Getting Job Description");

  h_Q->Front = (h_Q->Front+1)%(h_Q->Capacity);

  cudaMemcpyAsync(movePointer(Q, 16), movePointer(h_Q, 16), 
                   sizeof(int), cudaMemcpyHostToDevice, stream_dataOut);
  synchronizeAndPrint(stream_dataOut, "FandDJob, Updating Queue");

  free(h_Q);

  return *result;
}
