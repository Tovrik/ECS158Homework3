#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <string.h>
using namespace std;

__global__ void doComputations(float *x, float *y, float *m, int n, float h) {
  int blockIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

  float sum = 0;
  float count = 0;

  for(int i = 0; i < n; i++) {
    if(fabsf(x[blockIndex] - x[i]) < h) {
       sum = sum + y[i];
       count = count + 1;
    }
  }
  m[blockIndex] = sum / count;
}

void smoothc(float *x, float *y, float *m, int n, float h) {
  // Programatically determine the cuda Properties
  cudaDeviceProp Props;
  cudaGetDeviceProperties(&Props , 0);
  // Declare and allocate host and device memory
  // Device Memory pointers
  float *xchunk;
  float *ychunk;
  float *avgsPointer;

  int totalBlocks = 0;
  int threads_per_block = 0;
  double totalGlobalMemProgram = Props.totalGlobalMem / 2;
  int remaining = (24 * n) + 12;
  int chunks = ceil((float) remaining / totalGlobalMemProgram);
  int offset = 0;
  bool it_fits = false;

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

  // printf("TOTAL GLOBAL MEMORY = %d\n",Props.totalGlobalMem);
  // printf("chunks = %d\n", chunks);
  // printf("MAX THREADS CUDA = %d\n", Props.maxThreadsPerBlock);
  // printf("total blocks = %d\n", totalBlocks);
  // printf("threads/block = %d\n", threads_per_block);


  // Transfer the host arrays to Device
  while(1) {
  cudaMemcpy(xchunk, &x[offset], sizeof(float) * (n/chunks), cudaMemcpyHostToDevice);
  cudaMemcpy(ychunk, &y[offset], sizeof(float) * (n/chunks), cudaMemcpyHostToDevice);
  cudaMemcpy(avgsPointer, &m[offset], sizeof(float)* (n/chunks), cudaMemcpyHostToDevice);

  // Invoke the kernel
  doComputations <<<totalBlocks, threads_per_block>>> (xchunk, ychunk, avgsPointer, n, h);
  // Wait for kernel to finish()
  cudaThreadSynchronize();
  // Copy from device to host. 
  cudaMemcpy(&m[offset], avgsPointer, sizeof(float)*(n/chunks), cudaMemcpyDeviceToHost);

  offset += (n/chunks);
  if(offset >= n || it_fits)
    break;
  }

  // Free memory
  cudaFree(xchunk);
  cudaFree(ychunk);
  cudaFree(avgsPointer);
}



// int main(int argc, char** argv) {
//   // Host memory arrays
//   int n = 50000;
//   float h = 2;
//   // Allocate memory dynamically
//   float* x = new float[n];
//   float* y = new float[n];
//   float* averageArrays = new float[n]; 
//   //memset(averageArrays, 0, sizeof(averageArrays));
//   for(int i = 0, j = n; i < n; i++, j++) {
//     x[i] = i + 1;
//     y[i] = j + 1;
//   }
//   smoothc(x, y, averageArrays, n, h);

//   for(int i = 0; i < n; i++) {
//     cout << averageArrays[i] << "\n";
//   }

//   delete[] averageArrays;
//   delete[] y;
//   delete[] x;
//   return 0;
// }