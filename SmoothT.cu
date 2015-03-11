#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/count.h>
using namespace std;

struct finddiff {
  const float a;
  const float x2;

  finddiff(float _a, float _x2) : a(_a), x2(_x2) {}

  __device__ float operator()(const float& x1)
  {
    return (abs(x1 - x2) < a) ? 1: 0;
  }
};


void smootht(float *x, float *y, float *m, int n, float h) {
  // Alocate vector on device
  thrust:: device_vector<float> xchunk(x, x+n);
  thrust:: device_vector<float> ychunk(y, y+n);
  thrust:: device_vector<float> averageChunk(n);
  thrust:: device_vector<float> dv(n);

  for(int i = 0; i < n; i++) {
  
    thrust::transform(xchunk.begin(), xchunk.end(), dv.begin(), finddiff(h, xchunk[i]));

    // cout << "differences: " << endl;
    // thrust::copy(dv.begin(), dv.end(), std::ostream_iterator<float>(std::cout, "\n"));
    //cout << "xchunk[i] = " << xchunk[i] << endl;
    float sum = 0;
    // Count the number of 1s
    int count = thrust::count(dv.begin(), dv.end(), 1);

    // Go through the differences vector and find all elements of 1
    // And add those elements
    for(int j = 0; j < dv.size(); j++) {
      if(dv[j] == 1) {
        sum = sum + ychunk[j];
      }
    }
    averageChunk[i] = sum / count;
  }
  //cout << "averageChunk: " << endl;
  //thrust::copy(averageChunk.begin(), averageChunk.end(), std::ostream_iterator<float>(std::cout, "\n"));
  // Copy from device back to host
  thrust:: copy(averageChunk.begin(), averageChunk.end(), m);
}




int main(int argc, char** argv) {
  // Host memory arrays
  int n = 10;
  float h = 2;
  // Allocate memory dynamically
  float* x = new float[n];
  float* y = new float[n];
  float* averageArrays = new float[n]; 
  //memset(averageArrays, 0, sizeof(averageArrays));
  for(int i = 0, j = n; i < n; i++, j++) {
    x[i] = i + 1;
    y[i] = j + 1;
  }
  smootht(x, y, averageArrays, n, h);

  for(int i = 0; i < n; i++) {
     cout << averageArrays[i] << "\n";
  }

  delete[] averageArrays;
  delete[] y;
  delete[] x;
  return 0;
}
