#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
using namespace std;

__global__ void smootht(float *x, float *y, float *m, int n, float h) {
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

int main (int argc, char** argv) {
  cudaDeviceProp Props;
  cudaGetDeviceProperties(&Props , 0);

  int n = 10;
  float h = 3;
  // Host memory vectors
  thrust:: host_vector<float> x(n);
  thrust:: host_vector<float> y(n);
  thrust:: host_vector<float> averageArrays(n);

  // Populate the arrays
  for(int i = 0, j = n; i < n; i++, j++) {
    x[i] = i + 1;
    y[i] = j + 1;
  }

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


  // Print for testing purposes
  cout << "x: " << endl;
  thrust::copy(x.begin(), x.end(), std::ostream_iterator<float>(std::cout, "\n"));
  cout << "\n\ny: " << endl;
  thrust::copy(y.begin(), y.end(), std::ostream_iterator<float>(std::cout, "\n"));

  // Alocate vector on device
  thrust:: device_vector<float> xchunk = x;
  thrust:: device_vector<float> ychunk = y;
  thrust:: device_vector<float> averageChunk = averageArrays;

  float* xPointer = thrust::raw_pointer_cast( &xchunk[0] );
  float* yPointer = thrust::raw_pointer_cast( &ychunk[0] );
  float* avgPointer = thrust::raw_pointer_cast( &averageChunk[0] );

  // Invoke the kernel
  smootht <<< totalBlocks, threads_per_block >>> (xPointer, yPointer, avgPointer, n, h);

  cudaThreadSynchronize();

  // Copy from device back to host
  thrust:: copy(averageChunk.begin(), averageChunk.end(), averageArrays.begin());

  cout << "\n Printing averageArray " << endl;
  for(int i = 0; i < averageChunk.size(); i++) {
    cout << averageChunk[i] << endl;
  } 
}