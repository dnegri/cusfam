#include <stdio.h>
#include "pch.h"
#include "Geometry.h"
#include "NodalCuda.h"
#include "NodalCPU.h"

Geometry * g; 
NodalCuda * sanm2n;
NodalCPU* nodal_cpu;
CrossSection* xs;

float* sfam_jnet;
double* sfam_flux;
double sfam_reigv;


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
	sfam_jnet = new float[g->nsurf() * g->ng()];
	sfam_flux = new double[g->nxyz() * g->ng()];

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

	nodal_cpu = new NodalCPU(*g, *xs);
	nodal_cpu->init();
}

extern "C" void resetSANM2N(double* reigv_, double* jnet, double* phif)
{
	sfam_reigv = *reigv_;

	for (size_t ls = 0; ls < g->nsurf(); ls++)
	{
		int idirl = g->idirlr(LEFT, ls);
		int idirr = g->idirlr(RIGHT, ls);
		int lkl = g->lklr(LEFT, ls);
		int lkr = g->lklr(RIGHT, ls);
		int kl = lkl / g->nxy();
		int ll = lkl % g->nxy();
		int kr = lkr / g->nxy();
		int lr = lkr % g->nxy();


		for (size_t ig = 0; ig < g->ng(); ig++)
		{
			if (lkr < 0) {
				int idx =
					idirl * (g->nz() * g->nxy() * g->ng() * LR)
					+ kl * (g->nxy() * g->ng() * LR)
					+ ll * (g->ng() * LR)
					+ ig * LR
					+ RIGHT;
				sfam_jnet[ls*g->ng()+ig] = jnet[idx];
			}
			else {
				int idx =
					idirr * (g->nz() * g->nxy() * g->ng() * LR)
					+ kr * (g->nxy() * g->ng() * LR)
					+ lr * (g->ng() * LR)
					+ ig * LR
					+ LEFT;
				sfam_jnet[ls * g->ng() + ig] = jnet[idx];
			}
		}
	}

	int lk = -1;
	for (size_t k = 0; k < g->nz(); k++)
	{
		for (size_t l = 0; l < g->nxy(); l++)
		{
			lk++;
			for (size_t ig = 0; ig < g->ng(); ig++)
			{
				int idx = (k + 1) * (g->nxy() + 1) * g->ng() + (l + 1) * g->ng() + ig;
				sfam_flux[lk*g->ng()+ig] = phif[idx];
			}
		}
	}
	nodal_cpu->reset(*xs, sfam_reigv, sfam_jnet, sfam_flux);
	sanm2n->reset(*xs, sfam_reigv, sfam_jnet, sfam_flux);
}

extern "C" void runSANM2N(double* jnet)
{
	nodal_cpu->drive(sfam_jnet);
	sanm2n->drive(sfam_jnet);

	for (size_t ls = 0; ls < g->nsurf(); ls++)
	{
		int idirl = g->idirlr(LEFT, ls);
		int idirr = g->idirlr(RIGHT, ls);
		int lkl = g->lklr(LEFT, ls);
		int lkr = g->lklr(RIGHT, ls);
		int kl = lkl / g->nxy();
		int ll = lkl % g->nxy();
		int kr = lkr / g->nxy();
		int lr = lkr % g->nxy();

		for (size_t ig = 0; ig < g->ng(); ig++)
		{
			if (lkl < 0) {
				int idx =
					idirr * (g->nz() * g->nxy() * g->ng() * LR)
					+ kr * (g->nxy() * g->ng() * LR)
					+ lr * (g->ng() * LR)
					+ ig * LR
					+ LEFT;
				jnet[idx] = sfam_jnet[ls * g->ng() + ig];
			}else if (lkr < 0) {
				int idx =
					idirl * (g->nz() * g->nxy() * g->ng() * LR)
					+ kl * (g->nxy() * g->ng() * LR)
					+ ll * (g->ng() * LR)
					+ ig * LR
					+ RIGHT;
				jnet[idx] = sfam_jnet[ls * g->ng() + ig];
			} else {
				int idx =
					idirr * (g->nz() * g->nxy() * g->ng() * LR)
					+ kr * (g->nxy() * g->ng() * LR)
					+ lr * (g->ng() * LR)
					+ ig * LR
					+ LEFT;
				jnet[idx] = sfam_jnet[ls * g->ng() + ig];
				
				idx =
					idirl * (g->nz() * g->nxy() * g->ng() * LR)
					+ kl * (g->nxy() * g->ng() * LR)
					+ ll * (g->ng() * LR)
					+ ig * LR
					+ RIGHT;

				jnet[idx] = sfam_jnet[ls * g->ng() + ig];
			}
		}
	}
}