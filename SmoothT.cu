#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
using namespace std;


void smootht(float *x, float *y, float *m, int n, float h) {
  
}

int main (int argc, char** argv) {
  cudaDeviceProp Props;
  cudaGetDeviceProperties(&Props , 0);

  int n = 10;
  float h = 2;
  // Host memory vectors
  thrust:: host_vector<float> x(10);
  thrust:: host_vector<float> y(10);
  thrust:: host_vector<float> averageArrays(10);

  // Populate the arrays
  for(int i = 0, j = n; i < n; i++, j++) {
    x[i] = i + 1;
    y[i] = j + 1;
  }

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
  smootht <<< 1, 1 >>> (xPointer, yPointer, avgPointer, n, h);

  thrust:: copy(averageChunk.begin(), averageChunk.end(), averageArrays.begin());

  for(int i = 0; i < averageChunk.size(); i++) {
    cout << averageChunk[i] << endl;
  } 




}