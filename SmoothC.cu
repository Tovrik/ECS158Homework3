#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <string.h>
using namespace std;

__global__ void smoothc(float *x, float *y, float *m, int n, float h, int chunksize) {
  /*
  blockDim.x => gives the number of threads in a block, in the particular direction
  gridDim.x => gives the number of blocks in a grid
  */

  int blockIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

  //printf("threadIdx.x = %d\n", threadIdx.x);
  //printf("blockDim.x = %d\n", blockDim.x);

  float sum = 0;
  int count = 0;

  for(int i = 0; i < (n/chunksize); i++) {
    if(fabsf(x[blockIndex] - x[i]) < h) {
       //printf("x[blockIndex] = %d and x[i] = %d\n", x[blockIndex], x[i]);
       sum = sum + y[i];
       count = count + 1;
    }
  }
  //__syncthreads(); 
  //printf("sum = %f\n", sum);
  //printf("count = %d\n", count);
  m[blockIndex] = sum / count;
}

int main(int argc, char** argv) {
  cudaDeviceProp Props;
  cudaGetDeviceProperties(&Props , 0);
  // Declare and allocate host and device memory

  // Host memory arrays
  int n = 10000;
  float h = 5;
  float x[n];
  float y[n];
  float averageArrays[n];   
  memset(averageArrays, 0, sizeof(averageArrays));
  for(int i = 0, j = n; i < n; i++, j++) {
    x[i] = i + 1;
    y[i] = j + 1;
  }
  // printf("x[0] = %f\n", x[0]);
  // printf("x[2499] = %f\n", x[2499]);
  // printf("y[0] = %f\n", y[0]);
  // printf("y[2499] = %f\n", y[2499]);

  // Device Memory pointers
  float *xchunk;
  float *ychunk;
  float *avgsPointer;

  int totalBlocks = 0;
  int threads_per_block = 0;
  int totalSharedMemProgram = Props.totalGlobalMem / 2;
  int remaining = (24 * n) + 8;
  int chunks = ceil((float) remaining / totalSharedMemProgram);
  int offset = 0;
  bool it_fits = false;

  printf("chunks = %d\n", chunks);

  if (chunks == 1) it_fits = true;

  // Allocate memory on the device
  cudaMalloc((void**) &xchunk, sizeof(float) * (n/chunks) );
  cudaMalloc((void**) &ychunk, sizeof(float) * (n/chunks) );
  cudaMalloc((void**) &avgsPointer, sizeof(float) * (n/chunks) );

  if(n < (Props.maxThreadsPerBlock-1) ){
    totalBlocks = 1;
    threads_per_block = n;
  }
  else {
    totalBlocks = (int)(ceil((float)n / (Props.maxThreadsPerBlock/2) ));
    threads_per_block = Props.maxThreadsPerBlock / 2;
  }
  printf("MAX THREADS CUDA = %d\n", Props.maxThreadsPerBlock);
  printf("total blocks = %d\n", totalBlocks);
  printf("threads/block = %d\n", threads_per_block);


  // Transfer the host arrays to Device
  while(1) {
  cudaMemcpy(xchunk, &x[offset], sizeof(float) * (n/chunks), cudaMemcpyHostToDevice);
  cudaMemcpy(ychunk, &y[offset], sizeof(float) * (n/chunks), cudaMemcpyHostToDevice);
  cudaMemcpy(avgsPointer, &averageArrays[offset], sizeof(float)* (n/chunks), cudaMemcpyHostToDevice);

  // Invoke the kernel
  smoothc <<<totalBlocks, threads_per_block>>> (xchunk, ychunk, avgsPointer, n, h, chunks);
  // Wait for kernel to finish()
  cudaThreadSynchronize();
  // Copy from device to host. 
  cudaMemcpy(&averageArrays[offset], avgsPointer, sizeof(float)*(n/chunks), cudaMemcpyDeviceToHost);

  offset += (n/chunks);
  if(offset >= n || it_fits)
    break;
  }

  for(int i = 0; i < n; i++) {
    cout << averageArrays[i] << endl;
  }
  // Free memory
  cudaFree(xchunk);
  cudaFree(ychunk);
  cudaFree(avgsPointer);
  return 0;
}