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
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <cuda.h>

using namespace std;

typedef thrust::tuple<int, int>            tpl2int;
typedef thrust::device_vector<int>::iterator intiter;
typedef thrust::counting_iterator<int>     countiter;
typedef thrust::tuple<intiter, countiter>  tpl2intiter;
typedef thrust::zip_iterator<tpl2intiter>  idxzip;

// This functor returns true or false (1 or 0) based on 
// whether the abs(x1 - x2) is true or false 
struct finddiff {
  const float a;
  const float x2;

  finddiff(float _a, float _x2) : a(_a), x2(_x2) {}

  __device__ float operator()(const float& x1)
  {
    return (abs(x1 - x2) < a) ? 1: 0;
  }
};


struct select_unary_op : public thrust::unary_function<tpl2int, int>
{
  
  thrust::device_ptr<float> yptr;

  select_unary_op(thrust::device_ptr<float> _yptr) : yptr(_yptr) {}

  __device__ int operator()(const tpl2int& x) const
  {
    // If an element is true, then you get the y coordinate of the position
    if (x.get<0>() == 1)
      return yptr[x.get<1>()];
    else return 0;
   }
};

void smootht(float *x, float *y, float *m, int n, float h) {
  // Alocate vectors on device
  thrust:: device_vector<float> xchunk(x, x+n);
  thrust:: device_vector<float> ychunk(y, y+n);
  thrust:: device_vector<float> averageChunk(n);

  // You use these iterators with the zip iterators
  // to create a tuple
  thrust::counting_iterator<int> idxfirst(0);
  thrust::counting_iterator<int> idxlast = idxfirst + n;

  thrust:: device_vector<int> y_positions(n);
  thrust:: device_vector<int> dv(n);

  for(int i = 0; i < n; i++) {
    // Mark 1 in the array if abs(x1 - x2) is true, else 0
    thrust::transform(xchunk.begin(), xchunk.end(), dv.begin(), finddiff(h, xchunk[i]));
    // Count the number of 1's (times the condition was satisfied)
    int count = thrust::count(dv.begin(), dv.end(), 1);

    // Use a zip-iterator to make a tuple which has the following format: 
    // [position of element (1 or 0), index of element]
    idxzip first = thrust::make_zip_iterator(thrust::make_tuple(dv.begin(), idxfirst));
    idxzip last = thrust::make_zip_iterator(thrust::make_tuple(dv.end(), idxlast));
    thrust:: device_ptr<float> yPointer = &ychunk[0];

    // Pass the entire y_positions to my_unary_op 
    // and find the y values if the value is true, else put a 0
    select_unary_op my_unary_op(yPointer);
    thrust::transform(first, last, y_positions.begin(), my_unary_op);

    // Add all the values up to get the sum of y's
    // Don't have to worry about other positions because they hold a 0
    float sum = thrust::reduce(y_positions.begin(), y_positions.end());

    // Divide the sum by the count to find the average
    averageChunk[i] = sum / count;
  }
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
  for(int i = 0, j = n; i < n; i++, j++) {
    x[i] = i + 1;
    y[i] = j + 1;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);


  smootht(x, y, averageArrays, n, h);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime; 
  cudaEventElapsedTime(&elapsedTime , start, stop);

  for(int i = 0; i < n; i++) {
     cout << averageArrays[i] << "\n";
  }

  printf("Avg. time is %f ms\n", elapsedTime/100);

  delete[] averageArrays;
  delete[] y;
  delete[] x;
  return 0;
}
