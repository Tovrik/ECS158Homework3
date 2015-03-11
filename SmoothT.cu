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
using namespace std;

typedef thrust::tuple<int, int>            tpl2int;
typedef thrust::device_vector<int>::iterator intiter;
typedef thrust::counting_iterator<int>     countiter;
typedef thrust::tuple<intiter, countiter>  tpl2intiter;
typedef thrust::zip_iterator<tpl2intiter>  idxzip;

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
    if (x.get<0>() == 1)
      return yptr[x.get<1>()];
    else return 0;
   }
};

struct greater_than_negative_one {
  __device__
  bool operator()(int x) {
    return x != -1;
  }
};


void smootht(float *x, float *y, float *m, int n, float h) {
  // Alocate vector on device
  thrust:: device_vector<float> xchunk(x, x+n);
  thrust:: device_vector<float> ychunk(y, y+n);
  thrust:: device_vector<float> averageChunk(n);
  thrust:: device_vector<int> A(n);
  thrust:: sequence(A.begin(), A.end());
  thrust::counting_iterator<int> idxfirst(0);
  thrust::counting_iterator<int> idxlast = idxfirst + n;
  thrust:: device_vector<int> y_positions(n);
  thrust:: device_vector<int> dv(n);

  for(int i = 0; i < n; i++) {
  
    thrust::transform(xchunk.begin(), xchunk.end(), dv.begin(), finddiff(h, xchunk[i]));

    // cout << "differences: " << endl;
    // thrust::copy(dv.begin(), dv.end(), std::ostream_iterator<float>(std::cout, "\n"));
    //cout << "xchunk[i] = " << xchunk[i] << endl;
    // float sum = 0;
    // // Count the number of 1s
    int count = thrust::count(dv.begin(), dv.end(), 1);
    //cout << "Count = " << count << endl;

    // // Go through the differences vector and find all elements of 1
    // // And add those elements
    // for(int j = 0; j < dv.size(); j++) {
    //   if(dv[j] == 1) {
    //     sum = sum + ychunk[j];
    //   }
    // }
    // averageChunk[i] = sum / count;
    idxzip first = thrust::make_zip_iterator(thrust::make_tuple(dv.begin(), idxfirst));
    idxzip last = thrust::make_zip_iterator(thrust::make_tuple(dv.end(), idxlast));
    thrust:: device_ptr<float> yPointer = &ychunk[0];
    select_unary_op my_unary_op(yPointer);
    thrust::transform(first, last, y_positions.begin(), my_unary_op);
    float sum = thrust::reduce(y_positions.begin(), y_positions.end());
    // std::cout << "Results :" << std::endl;
    // thrust::copy(y_positions.begin(), y_positions.end(), std::ostream_iterator<int>( std::cout, " \n"));
    // cout << "Average = " << sum / count << endl;
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

  // for(int i = 0; i < n; i++) {
  //    cout << averageArrays[i] << "\n";
  // }

  delete[] averageArrays;
  delete[] y;
  delete[] x;
  return 0;
}
