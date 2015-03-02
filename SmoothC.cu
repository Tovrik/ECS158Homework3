#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <string.h>
using namespace std;

__global__ void smoothc(float *x, float *y, float *m, int n, float h) {
  int blockIndex = threadIdx.x;
  //printf("blockDim.x = %d\n", blockDim.x);
  float sum = 0;
  int count = 0;

  for(int i = 0; i < blockDim.x; i++) {
    if(fabsf(x[blockIndex] - x[i]) < h) {
       //printf("x[blockIndex] = %d and x[i] = %d\n", x[blockIndex], x[i]);
       sum = sum + y[i];
       count = count + 1;
    }
  }
  // __syncthreads();  
  m[blockIndex] = sum / count;
}

int main(int argc, char** argv) {
  // Declare and allocate host and device memory

  // Host memory arrays
  int n = 10;
  float h = 2;
  float x[n];
  float y[n];
  float averageArrays[n];   
  memset(averageArrays, 0, sizeof(averageArrays));
  for(int i = 0, j = n; i < n; i++, j++) {
    x[i] = i + 1;
    y[i] = j + 1;
  }

  // Device Memory pointers
  float *xchunk;
  float *ychunk;
  float *avgsPointer;

  // Allocate memory on the device
  cudaMalloc((void**) &xchunk, sizeof(float) * 100);
  cudaMalloc((void**) &ychunk, sizeof(float) * 100);
  cudaMalloc((void**) &avgsPointer, sizeof(float) * 100);

  // Transfer the host arrays to Device
  cudaMemcpy(xchunk, x, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(ychunk, y, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(avgsPointer, averageArrays, sizeof(float)* n, cudaMemcpyHostToDevice);

  // Set up Parameters for threads structure
  dim3 dimGrid(n, 1);
  dim3 dimBlock(1, 1, 1);

  // Invoke the kernel
  smoothc <<<1, n>>> (xchunk, ychunk, avgsPointer, n, h);
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