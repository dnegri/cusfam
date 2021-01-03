#include <cuda_runtime.h>
#include "Nodal.h"
#include "const.cuh"

#define d_jnet(lr,ig,l,k,idir)	(d_jnet[NDIRMAX*(d_nz*d_nxy*d_ng*LR)+k*(d_nxy*d_ng*LR)+l*(d_ng*LR)+ig*LR+lr])
#define d_trlcff0(ig,l,k,idir)	(d_ptr4(d_trlcff0,ig,l,k,idir) )

__global__ void calculateTransverseLeakage(float * d_jnet, float * d_trlcff0)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int k = tid / (d_ng * d_nxy);
	int l = tid / d_ng;
	int ig = tid % d_ng;

	float avgjnet[NDIRMAX], rh[NDIRMAX] = { 0.0 };

	avgjnet[XDIR] = (d_jnet(RIGHT, ig, l, k, XDIR) - d_jnet(LEFT, ig, l, k, XDIR)) * rh[XDIR];
	avgjnet[YDIR] = (d_jnet(RIGHT, ig, l, k, YDIR) - d_jnet(LEFT, ig, l, k, YDIR)) * rh[YDIR];
	avgjnet[ZDIR] = (d_jnet(RIGHT, ig, l, k, ZDIR) - d_jnet(LEFT, ig, l, k, ZDIR)) * rh[ZDIR];


	d_trlcff0(ig, l, k, XDIR) = avgjnet[YDIR] + avgjnet[ZDIR];
	d_trlcff0(ig, l, k, YDIR) = avgjnet[XDIR] + avgjnet[ZDIR];
	d_trlcff0(ig, l, k, ZDIR) = avgjnet[XDIR] + avgjnet[YDIR];
}

__global__ void calculateTransverseLeakage2(float* d_trlcff0, float* d_trlcff1, float* d_trlcff2)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int k = tid / (d_ng * d_nxy);
	int l = tid / d_ng;
	int ig = tid % d_ng;

	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
		//avgtrl3(:, CENTER) = d_trlcff0(:, lc, k, idir);
		//hmesh3(CENTER) = hmesh(idir, lc, k);

	}

}

Nodal::~Nodal()
{
}

void Nodal::drive()
{
	calculateTransverseLeakage();
}

void Nodal::calculateTransverseLeakage()
{
	dim3 blocks = dim3(_ng * _nxy * _nz / NTHREADSPERBLOCK + 1, 1, 1);
	dim3 threads = dim3(NTHREADSPERBLOCK, 1, 1);

	::calculateTransverseLeakage << <blocks, threads >> > (d_jnet, d_trlcff0);
}


