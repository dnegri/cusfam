#include "pch.h"
#include "NodalCuda.h"
#include "sanm2n.h"

static float* temp;

#define d_jnet(ig,lks)	    (d_jnet[lks*d_ng + ig])
#define d_trlcff0(ig,lkd)	(d_trlcff0[lkd*d_ng + ig])
#define d_trlcff1(ig,lkd)	(d_trlcff1[lkd*d_ng + ig])
#define d_trlcff2(ig,lkd)	(d_trlcff2[lkd*d_ng + ig])
#define d_eta1(ig,lkd)	(d_eta1[lkd*d_ng + ig])
#define d_eta2(ig,lkd)	(d_eta2[lkd*d_ng + ig])
#define d_m260(ig,lkd)	(d_m260[lkd*d_ng + ig])
#define d_m251(ig,lkd)	(d_m251[lkd*d_ng + ig])
#define d_m253(ig,lkd)	(d_m253[lkd*d_ng + ig])
#define d_m262(ig,lkd)	(d_m262[lkd*d_ng + ig])
#define d_m264(ig,lkd)	(d_m264[lkd*d_ng + ig])
#define d_diagD(ig,lkd)	(d_diagD[lkd*d_ng + ig])
#define d_diagDI(ig,lkd)	(d_diagDI[lkd*d_ng + ig])
#define d_mu(i,j,lkd)	(d_mu[lkd*d_ng2 + j*d_ng + i])
#define d_tau(i,j,lkd)	(d_tau[lkd*d_ng2 + j*d_ng + i])
#define d_matM(i,j,lk)	(d_matM[lk*d_ng2 + j*d_ng + i])
#define d_matMI(i,j,lk)	(d_matMI[lk*d_ng2 + j*d_ng + i])
#define d_matMs(i,j,lk)	(d_matMs[lk*d_ng2 + j*d_ng + i])
#define d_matMf(i,j,lk)	(d_matMf[lk*d_ng2 + j*d_ng + i])
#define d_xssf(i,j,lk)	(d_xssf[lk*d_ng2 + j*d_ng + i])
#define d_xsadf(ig,lk)	(d_xsadf[lk*d_ng + ig])
#define d_flux(ig,lk)	(d_flux[lk*d_ng+ig])

#define d_dsncff2(ig,lkd) (d_dsncff2[lkd*d_ng + ig])
#define d_dsncff4(ig,lkd) (d_dsncff4[lkd*d_ng + ig])
#define d_dsncff6(ig,lkd) (d_dsncff6[lkd*d_ng + ig])

#define d_hmesh(idir,lk)		(d_hmesh[lk*NDIRMAX+idir])
#define d_lktosfc(lr,idir,lk)	(d_lktosfc[(lk*NDIRMAX+idir)*LR + lr])
#define d_idirlr(lr,ls)			(d_idirlr[ls*LR + lr])
#define d_neib(lr, idir, lk)	(d_neib[(lk*NDIRMAX+idir)*LR + lr])
#define d_albedo(lr,idir)		(d_albedo[idir*LR + lr])

__global__ void reset(int& d_nxyz, float* d_hmesh, XS_PRECISION* d_xstf, XS_PRECISION* d_xsdf, float* d_eta1, float* d_eta2, float* d_m260, float* d_m251, float* d_m253, float* d_m262, float* d_m264, float* d_diagD, float* d_diagDI) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= d_nxyz) return;

	sanm2n_reset(lk, d_ng, d_ng2, d_nxyz, d_hmesh, d_xstf, d_xsdf, d_eta1, d_eta2, d_m260, d_m251, d_m253, d_m262, d_m264, d_diagD, d_diagDI);
}

__global__ void resetMatrix(int& d_nxyz, double& d_reigv, XS_PRECISION* d_xstf, XS_PRECISION* d_xsnff, XS_PRECISION* d_xschif, XS_PRECISION* d_xssf, float* d_matMs, float* d_matMf, float* d_matM) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;

	sanm2n_resetMatrix(lk, d_ng, d_ng2, d_nxyz, d_reigv, d_xstf, d_xsnff, d_xschif, d_xssf, d_matMs, d_matMf, d_matM);
}

__global__ void prepareMatrix(int& d_nxyz, float* d_m251, float* d_m253, float* d_diagD, float* d_diagDI, float* d_matM, float* d_matMI, float* d_tau, float* d_mu) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= d_nxyz) return;

	sanm2n_prepareMatrix(lk, d_ng, d_ng2, d_nxyz, d_m251, d_m253, d_diagD, d_diagDI, d_matM, d_matMI, d_tau, d_mu);
}

__global__ void calculateTransverseLeakage(int& d_nxyz, int* d_lktosfc, int* d_idirlr, int* d_neib, float* d_hmesh, float* d_albedo, float* d_jnet, float* d_trlcff0, float* d_trlcff1, float* d_trlcff2)
{
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= d_nxyz) return;

	sanm2n_calculateTransverseLeakage(lk, d_ng, d_ng2, d_nxyz, d_lktosfc, d_idirlr, d_neib, d_hmesh, d_albedo, d_jnet, d_trlcff0, d_trlcff1, d_trlcff2);

}

__global__ void calculateEven(int& d_nxyz, float* d_m260, float* d_m262, float* d_m264, float* d_diagD, float* d_diagDI, float* d_matM, double* d_flux, float* d_trlcff0, float* d_trlcff2, float* d_dsncff2, float* d_dsncff4, float* d_dsncff6)
{
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= d_nxyz) return;

	sanm2n_calculateEven(lk, d_ng, d_ng2, d_nxyz, d_m260, d_m262, d_m264, d_diagD, d_diagDI, d_matM, d_flux, d_trlcff0, d_trlcff2, d_dsncff2, d_dsncff4, d_dsncff6);
}

__global__ void calculateJnet(int& d_nsurf, int* d_lklr, int* d_idirlr, int* d_sgnlr, float* d_albedo, float* d_hmesh, XS_PRECISION* d_xsadf, float* d_m251, float* d_m253, float* d_m260, float* d_m262, float* d_m264, float* d_diagD, float* d_diagDI, float* d_matM, float* d_matMI, double* d_flux, float* d_trlcff0, float* d_trlcff1, float* d_trlcff2, float* d_mu, float* d_tau, float* d_eta1, float* d_eta2, float* d_dsncff2, float* d_dsncff4, float* d_dsncff6, float* d_jnet)
{
	int ls = threadIdx.x + blockIdx.x * blockDim.x;
	if (ls >= d_nsurf) return;

	::sanm2n_calculateJnet(ls, d_ng, d_ng2, d_nsurf, d_lklr, d_idirlr, d_sgnlr, d_albedo, d_hmesh, d_xsadf, d_m251, d_m253, d_m260, d_m262, d_m264, d_diagD, d_diagDI, d_matM, d_matMI, d_flux, d_trlcff0, d_trlcff1, d_trlcff2, d_mu, d_tau, d_eta1, d_eta2, d_dsncff2, d_dsncff4, d_dsncff6, d_jnet);

}

NodalCuda::NodalCuda(Geometry& g): Nodal(g)
{
	_ng = _g.ng();
	_ng2 = _ng * _ng;
	_nxyz = _g.nxyz();
	_nsurf = _g.nsurf();

	_d_blocks = dim3(_nxyz / NTHREADSPERBLOCK + 1, 1, 1);
	_d_threads = dim3(NTHREADSPERBLOCK, 1, 1);

	_d_blocks_sfc = dim3(_nsurf / NTHREADSPERBLOCK + 1, 1, 1);
	_d_threads_sfc = dim3(NTHREADSPERBLOCK, 1, 1);

	_jnet = new float[_nsurf * _ng];
	_flux = new double[_nxyz * _ng];


	checkCudaErrors(cudaMalloc((void**)&_d_symopt, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&_d_symang, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&_d_albedo, sizeof(float)*NDIRMAX*LR));
	checkCudaErrors(cudaMemcpy(_d_symopt, &_g.symopt(), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_symopt, &_g.symang(), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_albedo, &_g.albedo(0,0), sizeof(float) * NDIRMAX * LR, cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void**)&_d_nxyz, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&_d_nsurf, sizeof(int)));
	checkCudaErrors(cudaMemcpy(_d_nxyz, &_nxyz, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_nsurf, &_nsurf, sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&_d_neib, sizeof(int) * NEWSBT * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_d_lktosfc, sizeof(int) * NEWSBT * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_d_hmesh, sizeof(float) * NEWSBT * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_d_idirlr, sizeof(int) * LR* _nsurf));
	checkCudaErrors(cudaMalloc((void**)&_d_sgnlr, sizeof(int) * LR * _nsurf));
	checkCudaErrors(cudaMalloc((void**)&_d_lklr, sizeof(int) * LR * _nsurf));

	checkCudaErrors(cudaMemcpy(_d_neib	, &_g.neib(0, 0)	, sizeof(int) * NEWSBT * _nxyz	, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_lktosfc	, &_g.lktosfc(0,0,0), sizeof(int) * NEWSBT * _nxyz	, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_hmesh	, &_g.hmesh(0, 0)	, sizeof(float) * NEWSBT * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_idirlr	, &_g.idirlr(0, 0)	, sizeof(int) * LR * _nsurf		, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_sgnlr	, &_g.sgnlr(0, 0)	, sizeof(int) * LR * _nsurf		, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_lklr	, &_g.lklr(0, 0)	, sizeof(int) * LR * _nsurf		, cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void**)&_d_reigv, sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&_d_jnet, sizeof(float) * _nsurf * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_flux, sizeof(double) * _nxyz * _ng));

	checkCudaErrors(cudaMalloc((void**)&_d_trlcff0, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_trlcff1, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_trlcff2, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_eta1, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_eta2, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_mu, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_tau, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_m260, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_m251, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_m253, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_m262, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_m264, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_diagDI, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_diagD, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_dsncff2, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_dsncff4, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_dsncff6, sizeof(float) * _nxyz * NDIRMAX * _ng));

	checkCudaErrors(cudaMalloc((void**)&_d_xstf, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_xsdf, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_xsnff, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_xschif, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_xsadf, sizeof(XS_PRECISION) * _nxyz * _ng));

	checkCudaErrors(cudaMalloc((void**)&_d_xssf, sizeof(XS_PRECISION) * _nxyz * _ng2));
	checkCudaErrors(cudaMalloc((void**)&_d_matM, sizeof(float) * _nxyz * _ng2));
	checkCudaErrors(cudaMalloc((void**)&_d_matMI, sizeof(float) * _nxyz * _ng2));
	checkCudaErrors(cudaMalloc((void**)&_d_matMs, sizeof(float) * _nxyz * _ng2));
	checkCudaErrors(cudaMalloc((void**)&_d_matMf, sizeof(float) * _nxyz * _ng2));


}

NodalCuda::~NodalCuda()
{
}

void NodalCuda::init()
{
	temp = new float[10000];
}

void NodalCuda::reset(CrossSection& xs, double* reigv, double* jnet, double* phif)
{

	for (size_t ls = 0; ls < _nsurf; ls++)
	{
		int idirl = _g.idirlr(LEFT, ls);
		int idirr = _g.idirlr(RIGHT, ls);
		int lkl   = _g.lklr(LEFT, ls);
		int lkr   = _g.lklr(RIGHT, ls);
		int kl = lkl / _g.nxy();
		int ll = lkl % _g.nxy();
		int kr = lkr / _g.nxy();
		int lr = lkr % _g.nxy();


		for (size_t ig = 0; ig < _ng; ig++)
		{
			if (lkr < 0) {
				int idx =
					idirl * (_g.nz() * _g.nxy() * _g.ng() * LR)
					+ kl * (_g.nxy() * _g.ng() * LR)
					+ ll * (_g.ng() * LR)
					+ ig * LR
					+ RIGHT;
					this->jnet(ig, ls) = jnet[idx];
			}
			else {
				int idx =
					idirr * (_g.nz() * _g.nxy() * _g.ng() * LR)
					+ kr * (_g.nxy() * _g.ng() * LR)
					+ lr * (_g.ng() * LR)
					+ ig * LR
					+ LEFT;
				this->jnet(ig, ls) = jnet[idx];
			}
		}
	}

	int lk = -1;
	for (size_t k = 0; k < _g.nz(); k++)
	{
		for (size_t l = 0; l < _g.nxy(); l++)
		{
			lk++;
			for (size_t ig = 0; ig < _g.ng(); ig++)
			{
				int idx = (k + 1) *(_g.nxy()+1) * _g.ng() + (l + 1) * _g.ng() + ig;
				this->flux(ig, lk) = phif[idx];
			}
		}
	}

	_reigv = *reigv;
	checkCudaErrors(cudaMemcpy(_d_reigv, &_reigv, sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_flux, _flux, sizeof(double) * _nxyz * _ng, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(_d_xsnff, &xs.xsnf(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_xsdf, &xs.xsdf(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_xstf, &xs.xstf(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_xschif, &xs.chif(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_xsadf,&xs.xsadf(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_xssf, &xs.xssf(0, 0, 0), sizeof(XS_PRECISION) * _nxyz * _ng2, cudaMemcpyHostToDevice));

}

void NodalCuda::drive()
{
	::reset << <_d_blocks, _d_threads >> > (*_d_nxyz, _d_hmesh, _d_xstf, _d_xsdf, _d_eta1, _d_eta2, _d_m260, _d_m251, _d_m253, _d_m262, _d_m264, _d_diagD, _d_diagDI);
	::calculateTransverseLeakage << <_d_blocks, _d_threads >> > (*_d_nxyz, _d_lktosfc, _d_idirlr, _d_neib, _d_hmesh, _d_albedo, _d_jnet, _d_trlcff0, _d_trlcff1, _d_trlcff2);
	checkCudaErrors(cudaMemcpy(temp, _d_trlcff0, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _d_trlcff1, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _d_trlcff2, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));

	::resetMatrix << <_d_blocks, _d_threads >> > (*_d_nxyz, *_d_reigv, _d_xstf, _d_xsnff, _d_xschif, _d_xssf, _d_matMs, _d_matMf, _d_matM);

	checkCudaErrors(cudaMemcpy(temp, _d_matM, sizeof(float) * _nxyz * _ng2, cudaMemcpyDeviceToHost));

	::prepareMatrix << <_d_blocks, _d_threads >> > (*_d_nxyz, _d_m251, _d_m253, _d_diagD, _d_diagDI, _d_matM, _d_matMI, _d_tau, _d_mu);

	checkCudaErrors(cudaMemcpy(temp, _d_mu, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));


	::calculateEven << <_d_blocks, _d_threads >> > (*_d_nxyz, _d_m260, _d_m262, _d_m264, _d_diagD, _d_diagDI, _d_matM, _d_flux, _d_trlcff0, _d_trlcff2, _d_dsncff2, _d_dsncff4, _d_dsncff6);

	checkCudaErrors(cudaMemcpy(temp, _d_dsncff4, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _d_dsncff6, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _d_dsncff2, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));

	::calculateJnet << <_d_blocks_sfc, _d_threads_sfc >> > (*_d_nsurf, _d_lklr, _d_idirlr, _d_sgnlr, _d_albedo, _d_hmesh, _d_xsadf,_d_m251,_d_m253, _d_m260, _d_m262, _d_m264,
		_d_diagD, _d_diagDI, _d_matM, _d_matMI, _d_flux, _d_trlcff0, _d_trlcff1,
		_d_trlcff2, _d_mu, _d_tau, _d_eta1, _d_eta2, _d_dsncff2, _d_dsncff4, _d_dsncff6, _d_jnet);

	checkCudaErrors(cudaMemcpy(_jnet, _d_jnet,sizeof(float) * _nsurf * _ng, cudaMemcpyDeviceToHost));
}
