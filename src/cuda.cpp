#include <stdio.h>
#include "pch.h"
#include "Geometry.h"
#include "NodalCuda.h"
#include "NodalCPU.h"
#include "CMFDCPU.h"

Geometry * g; 
NodalCuda * sanm2n;
NodalCPU* nodal_cpu;
CrossSection* xs;

NODAL_PRECISION* sfam_jnet;
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


extern "C" void initCudaGeometry(int* ng, int* nxy, int* nz, int* nx, int* ny, int* nxs, int*nxe, 
							int* nys, int* nye, int* nsurf, int* ijtol, int* neibr, double* hmesh)
{
	g = new Geometry();
	g->init(ng, nxy, nz, nx, ny, nxs, nxe, nys, nye, nsurf, ijtol, neibr, hmesh);
	sfam_jnet = new NODAL_PRECISION[g->nsurf() * g->ng()];
	sfam_flux = new double[g->nxyz() * g->ng()];

}

extern "C" void initCudaXS(int* ng, int* nxy, int* nz, double*xsdf, double* xstf, double* xsnf, 
	double* xssf, double* xschif, double* xsadf)
{
	xs = new CrossSection(g->ng(), g->nxyz(), xsdf, xstf, xsnf, xssf, xschif, xsadf);
}

extern "C" void initCudaSolver()
{
	initCuda();


	sanm2n = new NodalCuda(*g);
	sanm2n->init();

	nodal_cpu = new NodalCPU(*g, *xs);
	nodal_cpu->init();
}

extern "C" void updateCuda(double* reigv_, double* jnet, double* phif)
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

extern "C" void runCuda(double* jnet)
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
		int sgnl = g->sgnlr(LEFT, ls);
		int sgnr = g->sgnlr(RIGHT, ls);

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
				jnet[idx] = sgnr * sfam_jnet[ls * g->ng() + ig];
				
				if (sgnl == - 1) {
					idx =
						idirl * (g->nz() * g->nxy() * g->ng() * LR)
						+ kl * (g->nxy() * g->ng() * LR)
						+ ll * (g->ng() * LR)
						+ ig * LR
						+ LEFT;
				}
				else {
					idx =
						idirl * (g->nz() * g->nxy() * g->ng() * LR)
						+ kl * (g->nxy() * g->ng() * LR)
						+ ll * (g->ng() * LR)
						+ ig * LR
						+ RIGHT;
				}


				jnet[idx] = sgnl * sfam_jnet[ls * g->ng() + ig];
			}
		}
	}
}

extern "C" void runCMFD(double* reigv_, double* psi, double* phif, double* jnet)
{
	float errl2=1.0;
	CMFDCPU cmfd(*g, *xs);
	cmfd.setNcmfd(5);
	cmfd.setEpsl2(1.0E-5);
	cmfd.setEshift(0.00);

	NodalCPU nodal(*g, *xs);
	cmfd.upddtil();
	for (int i = 0; i < 50; ++i) {
		cmfd.setls();
		cmfd.drive(*reigv_, phif, psi, errl2);
		cmfd.updjnet(phif, jnet);
		nodal.reset(*xs, *reigv_, jnet, phif);
		nodal.drive(jnet);
		cmfd.upddhat(phif, jnet);
		//if (i > 3 && errl2 < 1E-6) break;
	}

	fflush(stdout);
}