#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

using namespace std;


void smootht(float *x, float *y, float *m, int n, float h) {
	// qsort(x,n,sizeof(float),compare_floats);
	float sum = 0;
	float absVal = 0;
	int count = 0;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			 absVal = abs(x[i] - x[j]);
			 if(absVal < h){
			 	sum += y[j];
			 	count++;
			 }
		}
		m[i] = sum/count;
		sum = 0;
		count = 0;
	}
}

int main (int argc, char** argv) {
	float x[10] = {1,2,3,4,5,6,7,8,9,10};
	float y[10] = {11,12,13,14,15,16,17,18,19,20};
	float m[10];
	smootht(x,y,m,10,3);
	for(int i = 0; i < 10; i++)
		cout << m[i] << endl;
	return 0;

}