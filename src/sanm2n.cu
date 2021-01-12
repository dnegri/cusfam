#include <stdio.h>
#include "NodalCuda.h"
#include "helper_cuda.h"

Geometry * g;
NodalCuda * sanm2n;
CrossSection* xs;

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

extern "C" void setBoundaryCondtition(int* symopt, int* symang, double* albedo)
{
	g->setBoudnaryCondition(symopt, symang, albedo);
}


extern "C" void initGeometry(int* ng, int* nxy, int* nz, int* nx, int* ny, int* nxs, int*nxe, 
							int* nys, int* nye, int* nsurf, int* ijtol, int* neibr, double* hmesh)
{
	g = new Geometry();
	g->init(ng, nxy, nz, nx, ny, nxs, nxe, nys, nye, nsurf, ijtol, neibr, hmesh);
}

extern "C" void initCrossSection(int* ng, int* nxy, int* nz, double*xsdf, double* xstf, double* xsnf, 
	double* xssf, double* xschif, double* xsadf)
{
	xs = new CrossSection(g->ng(), g->nxyz(), xsdf, xstf, xsnf, xssf, xschif, xsadf);
}

extern "C" void initSANM2N()
{
	initCuda();
	sanm2n = new NodalCuda(*g);
	sanm2n->init();
}

extern "C" void resetSANM2N(double* reigv, double* jnet, double* phif)
{
	sanm2n->reset(*xs, reigv, jnet,phif);
}

extern "C" void runSANM2N()
{
	sanm2n->drive();
}