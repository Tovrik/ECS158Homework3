#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <cuda.h>
using namespace std;

#define GRID_SIZE 32
#define SHARED_MEM 16384

__global__ void findY(float *x, float *y, int n, float h, float z, int zLoc, float *returnVal) {
	// int col = blockIdx.x * blockDim.x + threadIdx.x;
	// int row = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float sum;
	sum = 0;
	// float absVal = 0;
	int count = 0;
	for(int i = 0; i < n; i++) {
		// absVal = abs(x[i] - z);
		if(abs(x[i] - z) < h) {
			//sum = atomicAdd(&sum, y[zLoc]);
		 	sum += y[i];
		 	// cuPrintf("sum = %d\n", sum);
		 	count++;
		}
	}
	*returnVal = sum / count;
	// sum = 0;
	// count = 0;
}

void smoothc(float *x, float *y, float *m, int n, float h) {
	// qsort(x,n,sizeof(float),compare_floats);
	dim3 dimGrid(1, 1);
	dim3 dimBlock(1, 1, 1);
	

	int chunksize = (SHARED_MEM / 2) - 32;

	float *xChunk;
	float *yChunk;
	float *mdev;
	//min size of x is 512 bytes or 64 entries
	int msize = chunksize * sizeof(float);
	cudaMalloc((void **) &xChunk, 80);
	cudaMalloc((void **) &yChunk, 80);
	cudaMalloc((void **) &mdev, 80);


	

	for(int i = 0; i < n; i++) {
		cudaMemcpy(xChunk, x, 80, cudaMemcpyHostToDevice);
		cudaMemcpy(yChunk, y, 80, cudaMemcpyHostToDevice);
		findY<<<dimGrid, dimBlock>>>(xChunk, yChunk, 10, h, x[i], i, &mdev[i]);
	}
	cudaMemcpy(m, mdev, 80, cudaMemcpyDeviceToHost);
	cudaFree(xChunk);
	cudaFree(yChunk);
	cudaFree(mdev);

}



int main (int argc, char** argv) {
	float x[10] = {1,2,3,4,5,6,7,8,9,10};
	float y[10] = {11,12,13,14,15,16,17,18,19,20};
	float m[10];
	for(int i = 0; i < 10; i++)
		cout << m[i] << endl;
	smoothc(x,y,m,10,3);
	cout<<"**********RETURN VALUES:***********"<<endl;
	for(int i = 0; i < 10; i++)
		cout << m[i] << endl;
	return 0;

}
