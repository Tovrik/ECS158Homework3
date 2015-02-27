ALL:
	/usr/local/cuda-5.5/bin/nvcc -g -G -o SmoothC.out SmoothC.cu -arch=sm_11
clean:
	rm -rf SmoothC.out

r:
	SmoothC.out

d:
	gdb SmoothC.out