#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <cuda.h>
using namespace std;

#define GRID_SIZE 32
#define SHARED_MEM 16384
#define HOST_SIZE 80
#define DEVICE_SIZE 80
#define SIZE 80

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
	//set up streams

	int nStreams = 4;
	cudaStream_t stream1, stream2, stream3, stream4;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);
	//create device memory
	float *xChunk1, *yChunk1, *mdev1, *xChunk2, *yChunk2, *mdev2, 
		*xChunk3, *yChunk3, *mdev3, *xChunk4, *yChunk4, *mdev4;
	cudaMalloc((void **) &xChunk1, DEVICE_SIZE);
	cudaMalloc((void **) &yChunk1, DEVICE_SIZE);
	cudaMalloc((void **) &mdev1, DEVICE_SIZE);
	cudaMalloc((void **) &xChunk2, DEVICE_SIZE);
	cudaMalloc((void **) &yChunk2, DEVICE_SIZE);
	cudaMalloc((void **) &mdev2, DEVICE_SIZE);
	cudaMalloc((void **) &xChunk3, DEVICE_SIZE);
	cudaMalloc((void **) &yChunk3, DEVICE_SIZE);
	cudaMalloc((void **) &mdev3, DEVICE_SIZE);
	cudaMalloc((void **) &xChunk4, DEVICE_SIZE);
	cudaMalloc((void **) &yChunk4, DEVICE_SIZE);
	cudaMalloc((void **) &mdev4, DEVICE_SIZE);		

	float *host1, *host2, *host3, *host4;
	cudaMallocHost(&host1, HOST_SIZE);
	cudaMallocHost(&host2, HOST_SIZE);
	cudaMallocHost(&host3, HOST_SIZE);
	cudaMallocHost(&host4, HOST_SIZE);



	int chunksize = (SHARED_MEM / 2) - 32;


	//min size of x is 512 bytes or 64 entries
	int msize = chunksize * sizeof(float);



	

	for(int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		cudaMemcpyAsync(xChunk1, x, SIZE, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(yChunk1, y, SIZE, cudaMemcpyHostToDevice, stream2);
		findY<<<dimGrid, dimBlock, 0, stream3>>>(xChunk1, yChunk1, 10, h, x[i], i, &mdev1[i]);
		cudaMemcpyAsync(m, mdev1, SIZE, cudaMemcpyDeviceToHost, stream4);

	}
	//destroy the streams
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
	cudaStreamDestroy(stream4);

	//free the alloc'ed space
	cudaFree(xChunk1);
	cudaFree(yChunk1);
	cudaFree(mdev1);
	cudaFree(xChunk2);
	cudaFree(yChunk2);
	cudaFree(mdev2);
	cudaFree(xChunk3);
	cudaFree(yChunk3);
	cudaFree(mdev3);
	cudaFree(xChunk4);
	cudaFree(yChunk4);
	cudaFree(mdev4);			

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
