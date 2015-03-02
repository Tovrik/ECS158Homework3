#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <string.h>
using namespace std;

__global__ void smoothc(float *x, float *y, float *m, int n, float h) {
	/*
	blockDim.x => gives the number of threads in a block, in the particular direction
	gridDim.x => gives the number of blocks in a grid
	*/

  int blockIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

  //printf("threadIdx.x = %d\n", threadIdx.x);
  //printf("blockDim.x = %d\n", blockDim.x);

  float sum = 0;
  int count = 0;

  for(int i = 0; i < n; i++) {
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
  cudaDeviceProp Props ;
  cudaGetDeviceProperties(&Props , 0) ;
  // Declare and allocate host and device memory

  // Host memory arrays
  int n = 2500;
  float h = 2;
  float x[n];
  float y[n];
  float averageArrays[n];   
  memset(averageArrays, 0, sizeof(averageArrays));
  for(int i = 1000, j = n; i < n+1000; i++, j++) {
    x[i-1000] = i + 1;
    y[i-1000] = j + 1;
  }
  // printf("x[0] = %f\n", x[0]);
  // printf("x[2499] = %f\n", x[2499]);
  // printf("y[0] = %f\n", y[0]);
  // printf("y[2499] = %f\n", y[2499]);

  // Device Memory pointers
  float *xchunk;
  float *ychunk;
  float *avgsPointer;

  // Allocate memory on the device
  cudaMalloc((void**) &xchunk, sizeof(float) * n);
  cudaMalloc((void**) &ychunk, sizeof(float) * n);
  cudaMalloc((void**) &avgsPointer, sizeof(float) * n);

  // Transfer the host arrays to Device
  cudaMemcpy(xchunk, x, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(ychunk, y, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(avgsPointer, averageArrays, sizeof(float)* n, cudaMemcpyHostToDevice);

  // Set up Parameters for threads structure
  // dim3 dimGrid(n, 1);
  // dim3 dimBlock(1, 1, 1);

  int totalBlocks = 0;
  int threads_per_block = 0;

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

  // Invoke the kernel
  smoothc <<<totalBlocks, threads_per_block>>> (xchunk, ychunk, avgsPointer, n, h);
  // Wait for kernel to finish()
  cudaThreadSynchronize();
  // Copy from device to host. 
  cudaMemcpy(averageArrays, avgsPointer, sizeof(float)*n, cudaMemcpyDeviceToHost);

  for(int i = 0; i < n; i++) {
    cout << averageArrays[i] << endl;
  }
  // Free memory
  cudaFree(xchunk);
  cudaFree(ychunk);
  cudaFree(avgsPointer);
  return 0;
}