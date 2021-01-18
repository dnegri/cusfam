#include "pch.h"
#include "NodalCuda.h"
//#include "sanm2n.h"

#define d_jnet(ig,lks)	    (d_jnet[(lks)*d_ng + ig])
#define d_trlcff0(ig,lkd)	(d_trlcff0[(lkd)*d_ng + ig])
#define d_trlcff1(ig,lkd)	(d_trlcff1[(lkd)*d_ng + ig])
#define d_trlcff2(ig,lkd)	(d_trlcff2[(lkd)*d_ng + ig])
#define d_eta1(ig,lkd)	(d_eta1[(lkd)*d_ng + ig])
#define d_eta2(ig,lkd)	(d_eta2[(lkd)*d_ng + ig])
#define d_m260(ig,lkd)	(d_m260[(lkd)*d_ng + ig])
#define d_m251(ig,lkd)	(d_m251[(lkd)*d_ng + ig])
#define d_m253(ig,lkd)	(d_m253[(lkd)*d_ng + ig])
#define d_m262(ig,lkd)	(d_m262[(lkd)*d_ng + ig])
#define d_m264(ig,lkd)	(d_m264[(lkd)*d_ng + ig])
#define d_diagD(ig,lkd)	(d_diagD[(lkd)*d_ng + ig])
#define d_diagDI(ig,lkd)	(d_diagDI[(lkd)*d_ng + ig])
#define d_mu(i,j,lkd)	(d_mu[(lkd)*d_ng2 + (j)*d_ng + i])
#define d_tau(i,j,lkd)	(d_tau[(lkd)*d_ng2 + (j)*d_ng + i])
#define d_matM(i,j,lk)	(d_matM[(lk)*d_ng2 + (j)*d_ng + i])
#define d_matMI(i,j,lk)	(d_matMI[(lk)*d_ng2 + (j)*d_ng + i])
#define d_matMs(i,j,lk)	(d_matMs[(lk)*d_ng2 + (j)*d_ng + i])
#define d_matMf(i,j,lk)	(d_matMf[(lk)*d_ng2 + (j)*d_ng + i])
#define d_xssf(i,j,lk)	(d_xssf[(lk)*d_ng2 + (j)*d_ng + i])
#define d_xsadf(ig,lk)	(d_xsadf[(lk)*d_ng + ig])
#define d_flux(ig,lk)	(d_flux[(lk)*d_ng+ig])

#define d_dsncff2(ig,lkd) (d_dsncff2[(lkd)*d_ng + ig])
#define d_dsncff4(ig,lkd) (d_dsncff4[(lkd)*d_ng + ig])
#define d_dsncff6(ig,lkd) (d_dsncff6[(lkd)*d_ng + ig])

#define d_hmesh(idir,lk)		(d_hmesh[(lk)*NDIRMAX+idir])
#define d_lktosfc(lr,idir,lk)	(d_lktosfc[((lk)*NDIRMAX+(idir))*LR + lr])
#define d_idirlr(lr,ls)			(d_idirlr[(ls)*LR + lr])
#define d_neib(lr, idir, lk)	(d_neib[((lk)*NDIRMAX+(idir))*LR + lr])
#define d_albedo(lr,idir)		(d_albedo[(idir)*LR + lr])

__global__ void updateConstant(NodalCuda * nodal) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= *nodal->nxyz1()) return;

	//nodal.updateConstant(lk);
}

__global__ void updateMatrix(NodalCuda& nodal) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= nodal.nxyz()) return;

	nodal.updateMatrix(lk);
}

__global__ void calculateTransverseLeakage(NodalCuda& nodal)
{
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= nodal.nxyz()) return;

	nodal.calculateTransverseLeakage(lk);

}

__global__ void calculateEven(NodalCuda& nodal)
{
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= nodal.nxyz()) return;

	nodal.calculateEven(lk);
}

__global__ void calculateJnet(NodalCuda& nodal)
{
	int ls = threadIdx.x + blockIdx.x * blockDim.x;
	if (ls >= nodal.nsurf()) return;

	nodal.calculateJnet(ls);

}

__device__ __host__ NodalCuda::NodalCuda(Geometry& g): Nodal(g)
{
	_blocks = dim3(g.nxyz() / NTHREADSPERBLOCK + 1, 1, 1);
	_threads = dim3(NTHREADSPERBLOCK, 1, 1);
	_blocks_sfc = dim3(g.nsurf() / NTHREADSPERBLOCK + 1, 1, 1);
	_threads_sfc = dim3(NTHREADSPERBLOCK, 1, 1);

	_host_jnet = new float[g.nsurf() * g.ng()];
	_host_flux = new double[g.nxyz() * g.ng()];

	checkCudaErrors(cudaMalloc((void**)&_ng, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&_ng2, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&_nxyz, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&_nsurf, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&_symopt, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&_symang, sizeof(int)));

	checkCudaErrors(cudaMemcpy(_ng, &g.ng(), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_ng2, &g.ng2(), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_nxyz, &g.nxyz(), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_nsurf, &g.nsurf(), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_symopt, &g.symopt(), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_symang, &g.symang(), sizeof(int), cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void**)&_neib, sizeof(int) * NEWSBT * g.nxyz()));
	checkCudaErrors(cudaMalloc((void**)&_lktosfc, sizeof(int) * NEWSBT * g.nxyz()));
	checkCudaErrors(cudaMalloc((void**)&_hmesh, sizeof(float) * NEWSBT * g.nxyz()));
	checkCudaErrors(cudaMalloc((void**)&_idirlr, sizeof(int) * LR * g.nsurf()));
	checkCudaErrors(cudaMalloc((void**)&_sgnlr, sizeof(int) * LR * g.nsurf()));
	checkCudaErrors(cudaMalloc((void**)&_lklr, sizeof(int) * LR * g.nsurf()));

	checkCudaErrors(cudaMemcpy(_neib, &g.neib(0, 0), sizeof(int) * NEWSBT * g.nxyz(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_lktosfc, &g.lktosfc(0, 0, 0), sizeof(int) * NEWSBT * g.nxyz(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hmesh, &g.hmesh(0, 0), sizeof(float) * NEWSBT * g.nxyz(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_idirlr, &g.idirlr(0, 0), sizeof(int) * LR * g.nsurf(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_sgnlr, &g.sgnlr(0, 0), sizeof(int) * LR * g.nsurf(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_lklr, &g.lklr(0, 0), sizeof(int) * LR * g.nsurf(), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&_reigv, sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&_jnet, sizeof(float) * g.nsurf() * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_flux, sizeof(double) * g.nxyz() * g.ng()));

	checkCudaErrors(cudaMalloc((void**)&_trlcff0, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_trlcff1, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_trlcff2, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_eta1, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_eta2, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_m260, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_m251, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_m253, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_m262, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_m264, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_diagDI, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_diagD, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_dsncff2, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_dsncff4, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_dsncff6, sizeof(float) * g.nxyz() * NDIRMAX * g.ng()));

	checkCudaErrors(cudaMalloc((void**)&_xstf, sizeof(XS_PRECISION) * g.nxyz() * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_xsdf, sizeof(XS_PRECISION) * g.nxyz() * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_xsnf, sizeof(XS_PRECISION) * g.nxyz() * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_chif, sizeof(XS_PRECISION) * g.nxyz() * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_xsadf, sizeof(XS_PRECISION) * g.nxyz() * g.ng()));

	checkCudaErrors(cudaMalloc((void**)&_xssf, sizeof(XS_PRECISION) * g.nxyz() * g.ng2()));
	checkCudaErrors(cudaMalloc((void**)&_mu, sizeof(float) * g.nxyz() * NDIRMAX * g.ng2()));
	checkCudaErrors(cudaMalloc((void**)&_tau, sizeof(float) * g.nxyz() * NDIRMAX * g.ng2()));
	checkCudaErrors(cudaMalloc((void**)&_matM, sizeof(float) * g.nxyz() * g.ng2()));
	checkCudaErrors(cudaMalloc((void**)&_matMI, sizeof(float) * g.nxyz() * g.ng2()));
	checkCudaErrors(cudaMalloc((void**)&_matMs, sizeof(float) * g.nxyz() * g.ng2()));
	checkCudaErrors(cudaMalloc((void**)&_matMf, sizeof(float) * g.nxyz() * g.ng2()));
}

__device__ __host__ NodalCuda::~NodalCuda()
{
}

void NodalCuda::init()
{
}

void NodalCuda::reset(CrossSection& xs, double* reigv, double* jnet, double* phif)
{

	for (size_t ls = 0; ls < _g.nsurf(); ls++)
	{
		int idirl = _g.idirlr(LEFT, ls);
		int idirr = _g.idirlr(RIGHT, ls);
		int lkl = _g.lklr(LEFT, ls);
		int lkr = _g.lklr(RIGHT, ls);
		int kl = lkl / _g.nxy();
		int ll = lkl % _g.nxy();
		int kr = lkr / _g.nxy();
		int lr = lkr % _g.nxy();


		for (size_t ig = 0; ig < _g.ng(); ig++)
		{
			if (lkr < 0) {
				int idx =
					idirl * (_g.nz() * _g.nxy() * _g.ng() * LR)
					+ kl * (_g.nxy() * _g.ng() * LR)
					+ ll * (_g.ng() * LR)
					+ ig * LR
					+ RIGHT;
				this->host_jnet(ig, ls) = jnet[idx];
			}
			else {
				int idx =
					idirr * (_g.nz() * _g.nxy() * _g.ng() * LR)
					+ kr * (_g.nxy() * _g.ng() * LR)
					+ lr * (_g.ng() * LR)
					+ ig * LR
					+ LEFT;
				this->host_jnet(ig, ls) = jnet[idx];
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
				int idx = (k + 1) * (_g.nxy() + 1) * _g.ng() + (l + 1) * _g.ng() + ig;
				this->host_flux(ig, lk) = phif[idx];
			}
		}
	}

	_host_reigv = *reigv;
	_reigv = _host_reigv;
	checkCudaErrors(cudaMemcpy(_flux, _host_flux, sizeof(double) * _g.nxyz() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_jnet, _host_jnet, sizeof(float) * _g.nsurf() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsnf, &xs.xsnf(0, 0), sizeof(XS_PRECISION) * _g.nxyz() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsdf, &xs.xsdf(0, 0), sizeof(XS_PRECISION) * _g.nxyz() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xstf, &xs.xstf(0, 0), sizeof(XS_PRECISION) * _g.nxyz() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_chif, &xs.chif(0, 0), sizeof(XS_PRECISION) * _g.nxyz() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsadf, &xs.xsadf(0, 0), sizeof(XS_PRECISION) * _g.nxyz() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xssf, &xs.xssf(0, 0, 0), sizeof(XS_PRECISION) * _g.nxyz() * _g.ng2(), cudaMemcpyHostToDevice));
}

void NodalCuda::drive()
{
#ifdef _DEBUG
	float* temp = new float[_g.nxyz() * NDIRMAX * _g.ng()]{};
	checkCudaErrors(cudaMemcpy(temp, _trlcff0, sizeof(float) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _trlcff1, sizeof(float) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _trlcff2, sizeof(float) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _matM, sizeof(float) * _g.nxyz() * _g.ng2(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _dsncff4, sizeof(float) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _dsncff6, sizeof(float) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _dsncff2, sizeof(float) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(_host_jnet, _jnet, sizeof(float) * _g.nsurf() * _g.ng(), cudaMemcpyDeviceToHost));
	delete[] temp;
#endif

	::updateConstant << <1, 1 >> > (this);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(_host_jnet, _jnet, sizeof(float) * _g.nsurf() * _g.ng(), cudaMemcpyDeviceToHost));

	::calculateTransverseLeakage << <_blocks, _threads >> > (*this);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(_host_jnet, _jnet, sizeof(float) * _g.nsurf() * _g.ng(), cudaMemcpyDeviceToHost));

	::updateMatrix << <_blocks, _threads >> > (*this);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(_host_jnet, _jnet, sizeof(float) * _g.nsurf() * _g.ng(), cudaMemcpyDeviceToHost));


	::calculateEven << <_blocks, _threads >> > (*this);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(_host_jnet, _jnet, sizeof(float) * _g.nsurf() * _g.ng(), cudaMemcpyDeviceToHost));


	::calculateJnet << <_blocks_sfc, _threads_sfc >> > (*this);
	cudaDeviceSynchronize();

	checkCudaErrors(cudaMemcpy(_host_jnet, _jnet, sizeof(float) * _g.nsurf() * _g.ng(), cudaMemcpyDeviceToHost));

#ifdef _DEBUG
	//float* temp = new float[_g.nxyz() * NDIRMAX * _g.ng()]{};
	checkCudaErrors(cudaMemcpy(temp, _trlcff0, sizeof(float) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _trlcff1, sizeof(float) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _trlcff2, sizeof(float) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _matM, sizeof(float) * _g.nxyz() * _g.ng2(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _dsncff4, sizeof(float) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _dsncff6, sizeof(float) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _dsncff2, sizeof(float) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	delete[] temp;
#endif
}
