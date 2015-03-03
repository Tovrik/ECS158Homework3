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
	// Host memory vectors
	thrust:: host_vector<float> x(10);
	thrust:: host_vector<float> y(10);
	thrust:: host_vector<float> averageArrays(10);

	for(int i = 0, j = n; i < n; i++, j++) {
    x.push_back(i + 1);
    y.push_back(j + 1);
  }

  

}