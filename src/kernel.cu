#include <stdio.h>
#include "Nodal.h"
#include "helper_cuda.h"


Nodal sanm2n;

__global__ void initGeometry(int* d_neib) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	d_neib(LEFT , XDIR, lk) = 1;
	d_neib(RIGHT, XDIR, lk) = 1;
	d_neib(LEFT , YDIR, lk) = 1;
	d_neib(RIGHT, YDIR, lk) = 1;
	d_neib(LEFT , ZDIR, lk) = 1;
	d_neib(RIGHT, ZDIR, lk) = 1;
}

void initCuda() {
	cudaDeviceProp deviceProp;
	int devID = 0;
	printf("GPU selected Device ID = %d \n", devID);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	/* Statistics about the GPU device */
	printf("> GPU device has %d Multi-Processors, "
		"SM %d.%d compute capabilities\n\n",
		deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
}

extern "C" void initNodal(int* ng, int* nxy, int* nz, int* nsurf, int* neibr, double* hmesh)
{
	initCuda();

	sanm2n.d_nxy = *nxy;
	sanm2n.d_nxyz = *nxy * *nz;
	sanm2n.d_nsurf = *nsurf * *nz + (*nz + 1) * *nxy;

	int* neib = new int[]


	sanm2n.init();
	checkCudaErrors(cudaMalloc((void**)&sanm2n.d_neib, sizeof(int) * NEWSBT * sanm2n.d_nxyz));

	initGeometry<<<sanm2n._blocks,sanm2n._threads>>>(sanm2n.d_neib);
}