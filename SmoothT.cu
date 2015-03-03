#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
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


void smootht(float *x, float *y, float *m, int n, float h) {
  cudaDeviceProp Props;
  cudaGetDeviceProperties(&Props , 0);

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

  // printf("MAX THREADS CUDA = %d\n", Props.maxThreadsPerBlock);
  // printf("total blocks = %d\n", totalBlocks);
  // printf("threads/block = %d\n", threads_per_block);

  // Print for testing purposes
  // cout << "x: " << endl;
  // thrust::copy(x.begin(), x.end(), std::ostream_iterator<float>(std::cout, "\n"));
  // cout << "\n\ny: " << endl;
  // thrust::copy(y.begin(), y.end(), std::ostream_iterator<float>(std::cout, "\n"));

  // Alocate vector on device
  thrust:: device_vector<float> xchunk(x, x+n);
  thrust:: device_vector<float> ychunk(y, y+n);
  thrust:: device_vector<float> averageChunk(m, m+n);

  float* xPointer = thrust::raw_pointer_cast( &xchunk[0] );
  float* yPointer = thrust::raw_pointer_cast( &ychunk[0] );
  float* avgPointer = thrust::raw_pointer_cast( &averageChunk[0] ); 

  // Invoke the kernel
  doComputations <<< totalBlocks, threads_per_block >>> (xPointer, yPointer, avgPointer, n, h);

  cudaThreadSynchronize();

  // Copy from device back to host
  thrust:: copy(averageChunk.begin(), averageChunk.end(), m);
}



// int main (int argc, char** argv) {
//   int n = 50000;
//   float h = 2;
//   // Host memory vectors
//   thrust:: host_vector<float> xV(n);
//   thrust:: host_vector<float> yV(n);
//   thrust:: host_vector<float> averageArraysV(n);

//   // Populate the arrays
//   for(int i = 0, j = n; i < n; i++, j++) {
//     xV[i] = i + 1;
//     yV[i] = j + 1;
//   }

//   float* x = thrust::raw_pointer_cast( &xV[0] );
//   float* y = thrust::raw_pointer_cast( &yV[0] );
//   float* m = thrust::raw_pointer_cast( &averageArraysV[0] );

//   smootht(x, y, m, n, h);

//   cout << "\n\nAVERAGE ARRAYS CONTENTS: " << endl;
//   for(int i = 0; i < n; i++) {
//     cout << m[i] << endl;
//   }
//   return 0;
// }