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
  int n = 10;
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

}