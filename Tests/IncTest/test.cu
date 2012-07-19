#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "Kernels/incSuperKernel.cu"

#include <pthread.h>

/////////////////////////////////////////////////////////////////
// Global Variables
/////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////
// The Main
////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  cudaStream_t stream_kernel, stream_dataIn, stream_dataOut;
  cudaStreamCreate(&stream_kernel);
  cudaStreamCreate(&stream_dataIn);
  cudaStreamCreate(&stream_dataOut);  //currently these arent used



  int size = 5;

  int* h_init = (int*)malloc((size+1)*sizeof(int));
  int* h_result = (int*)malloc((size+1)*sizeof(int));

  int* d_init;
  cudaMalloc(&d_init, (size+1)*sizeof(int));
  int* d_result;
  cudaMalloc(&d_result, (size+1)*sizeof(int));

  h_init[0]=1;  //set the data ready flag to false
  cudaMemcpy(d_init, h_init, sizeof(int), cudaMemcpyHostToDevice);

  h_result[0]=1;  //set the data ready flag to false
  cudaMemcpy(d_result, h_result, sizeof(int), cudaMemcpyHostToDevice);

  dim3 threads(32, 1);
  dim3 grid(1, 1);

  printf("launching SuperKernel\n");

  // call the cudaMatrixMul cuda function
  superKernel<<< grid, threads, 0, stream_kernel>>>(d_init, size, d_result);

  int j;
  for(j=0;j<size;j++)h_init[j+1] = j;

  cudaMemcpy(&d_init[1], &h_init[1], size*sizeof(int), cudaMemcpyHostToDevice);

  h_init[0]=0;
  cudaMemcpy(d_init, h_init, sizeof(int), cudaMemcpyHostToDevice);

  int done = 1;
  while(done!=0) { cudaMemcpy(&done, d_result, sizeof(int), cudaMemcpyDeviceToHost); printf("got value done: %d\n", done); }

  cudaMemcpy(&h_result[1], &d_result[1], size*sizeof(int), cudaMemcpyDeviceToHost);

  int i;
  for(i=0; i<size; i++) printf("intial value: %d\t final value: %d\n", h_init[i+1], h_result[i+1]);

  return 0;    
}







