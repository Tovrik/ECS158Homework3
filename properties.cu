#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <string.h>
using namespace std;

void DisplayHeader()
{
   cudaDeviceProp Props ;
   cudaGetDeviceProperties(&Props , 0) ;
   printf("shared mem: %d\n", Props.sharedMemPerBlock);
   printf("max threads: %d\n", Props.maxThreadsPerBlock);
   printf("max blocks: %d\n", Props.maxGridSize[0]);
   printf("total Const mem: %d\n", Props.totalConstMem);
   printf("Major number: %d\n", Props.major);
   printf("Minor number: %d\n", Props.minor);
   printf("Total Global Memory: %d\n", Props.totalGlobalMem);
}

int main(int argc, char** argv) {
  DisplayHeader();


}