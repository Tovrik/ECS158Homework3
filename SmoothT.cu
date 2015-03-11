#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
using namespace std;

struct finddiff {
  __device__ float operator()(const float& x1, const float& x2)
  {
    return abs(x1 - x2);
  }
};



void smootht(float *x, float *y, float *m, int n, float h) {
  // Alocate vector on device
  thrust:: device_vector<float> xchunk(x, x+n);
  thrust:: device_vector<float> ychunk(y, y+n);
  thrust:: device_vector<float> averageChunk(m, m+n);
  thrust:: device_vector<float> dv(n);

  cout << "x: " << endl;
  thrust::copy(xchunk.begin(), xchunk.end(), std::ostream_iterator<float>(std::cout, "\n"));

  thrust:: device_vector<float> temp(n);
  thrust:: fill(temp.begin(), temp.end(), x[0]);
  thrust::transform(xchunk.begin(), xchunk.end(), temp.begin(), dv.begin(), finddiff());

  cout << "differences: " << endl;
  thrust::copy(dv.begin(), dv.end(), std::ostream_iterator<float>(std::cout, "\n"));


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

  // for(int i = 0; i < n; i++) {
  //   cout << averageArrays[i] << "\n";
  // }

  delete[] averageArrays;
  delete[] y;
  delete[] x;
  return 0;
}
