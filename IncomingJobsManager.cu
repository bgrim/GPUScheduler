#include <stdio.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "Kernels/SuperKernel.cu"
#include "Queues/QueueJobs.cu"
#include "Queues/QueueResults.cu"


pthread_t start_IncomingJobsManager(QueueResults results)
{
//This should do any initializing that the incoming jobs
//  manager will need and then launch a thread running
//  main_IncomingJobsManager(  ), returning that thread



}


void *main_IncomingJobsManager(void *params)
{
//The thread should get job descriptions some how and Enqueue them
//  into the queue in params
//    --eventually this should get jobs from an external application
//      but will probably just be hardcoded at first or a parameter



}

