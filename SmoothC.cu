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
	float absVal = 0;
	int count = 0;
	for(int i = 0; i < n; i++) {
		absVal = abs(x[i] - z);
		if(absVal < h) {
			atomicAdd(&sum, y[zLoc]);
		 	sum += y[zLoc];
		 	count++;
		}
	}
		*returnVal = sum/count;
		sum = 0;
		count = 0;
}

void smoothc(float *x, float *y, float *m, int n, float h) {
	// qsort(x,n,sizeof(float),compare_floats);
	dim3 dimGrid(1, 1);
	dim3 dimBlock(1, 1, 1);
	
	// Size of x and y: n / SHARED_MEM / 2 = 
	// The rest of the params: - 32
	// Combined this equals 16384 bytes
	int chunksize = (SHARED_MEM / 2) - 32;

	float *xChunk;
	float *yChunk;
	//min size of x is 512 bytes or 64 entries
	int msize = chunksize * sizeof(float);
	cudaMalloc((void **) &xChunk, msize);
	cudaMalloc((void **) &yChunk, msize);
	// dst 		- Destination memory address
	// src 		- Source memory address
	// count 	- Size in bytes to copy
	// kind 	- Type of transfer
	// cudaMemcpy(dst, src, count, kind);

	

	for(int i = 0; i < n; i++) {
		cudaMemcpy(xChunk, x, msize, cudaMemcpyHostToDevice);
		cudaMemcpy(yChunk, y, msize, cudaMemcpyHostToDevice);
		findY<<<dimGrid, dimBlock>>>(xChunk, yChunk, chunksize, h, x[i], i, &m[i]);
	}

	cudaFree(xChunk);
	cudaFree(yChunk);

}



int main (int argc, char** argv) {
	float x[10] = {1,2,3,4,5,6,7,8,9,10};
	float y[10] = {11,12,13,14,15,16,17,18,19,20};
	float m[10];
	smoothc(x,y,m,10,3);
	for(int i = 0; i < 10; i++)
		cout << m[i] << endl;
	return 0;

}
