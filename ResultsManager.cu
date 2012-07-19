#include <stdio.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "Kernels/SuperKernel.cu"
#include "Queues/QueueJobs.cu"
#include "Queues/QueueResults.cu"


pthread_t start_ResultsManager(QueueResults results)
{
//This should do any initializing that the results manager will
//  need and then launch a thread running main_ResultsManager,
//  returning this thread



}


void *main_ResultsManager(void *params)
{
//The thread should read results from the queue in param and
//  print something about them to the screen.
//    --eventually this should return the result to the application
//      that requested the work.



}

