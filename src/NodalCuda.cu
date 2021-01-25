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

__global__ void updateConstant(NodalCuda& nodal) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= nodal.nxyz()) return;

	nodal.updateConstant(lk);
}

__global__ void updateMatrix(NodalCuda& nodal) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= nodal.nxyz()) return;

	nodal.updateMatrix(lk);
}

__global__ void caltrlcff0(NodalCuda& nodal)
{
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= nodal.nxyz()) return;

	nodal.caltrlcff0(lk);

}
__global__ void caltrlcff12(NodalCuda& nodal)
{
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= nodal.nxyz()) return;

	nodal.caltrlcff12(lk);

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

__host__ NodalCuda::NodalCuda(Geometry& g): Nodal(g)
{
	_blocks = dim3(g.nxyz() / NTHREADSPERBLOCK + 1, 1, 1);
	_threads = dim3(NTHREADSPERBLOCK, 1, 1);
	_blocks_sfc = dim3(g.nsurf() / NTHREADSPERBLOCK + 1, 1, 1);
	_threads_sfc = dim3(NTHREADSPERBLOCK, 1, 1);


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
	checkCudaErrors(cudaMalloc((void**)&_albedo, sizeof(float) * NDIRMAX * LR));

	checkCudaErrors(cudaMemcpy(_neib, &g.neib(0, 0), sizeof(int) * NEWSBT * g.nxyz(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_lktosfc, &g.lktosfc(0, 0, 0), sizeof(int) * NEWSBT * g.nxyz(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hmesh, &g.hmesh(0, 0), sizeof(float) * NEWSBT * g.nxyz(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_idirlr, &g.idirlr(0, 0), sizeof(int) * LR * g.nsurf(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_sgnlr, &g.sgnlr(0, 0), sizeof(int) * LR * g.nsurf(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_lklr, &g.lklr(0, 0), sizeof(int) * LR * g.nsurf(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_albedo, &g.albedo(0, 0), sizeof(float) * NDIRMAX * LR, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&_reigv, sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&_flux, sizeof(double) * g.nxyz() * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_jnet, sizeof(NODAL_PRECISION) * g.nsurf() * g.ng()));

	checkCudaErrors(cudaMalloc((void**)&_trlcff0, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_trlcff1, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_trlcff2, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_eta1, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_eta2, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_m260, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_m251, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_m253, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_m262, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_m264, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_diagDI, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_diagD, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_dsncff2, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_dsncff4, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_dsncff6, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng()));

	checkCudaErrors(cudaMalloc((void**)&_xstf, sizeof(XS_PRECISION) * g.nxyz() * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_xsdf, sizeof(XS_PRECISION) * g.nxyz() * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_xsnf, sizeof(XS_PRECISION) * g.nxyz() * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_chif, sizeof(XS_PRECISION) * g.nxyz() * g.ng()));
	checkCudaErrors(cudaMalloc((void**)&_xsadf, sizeof(XS_PRECISION) * g.nxyz() * g.ng()));

	checkCudaErrors(cudaMalloc((void**)&_xssf, sizeof(XS_PRECISION) * g.nxyz() * g.ng2()));
	checkCudaErrors(cudaMalloc((void**)&_mu, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng2()));
	checkCudaErrors(cudaMalloc((void**)&_tau, sizeof(NODAL_PRECISION) * g.nxyz() * NDIRMAX * g.ng2()));
	checkCudaErrors(cudaMalloc((void**)&_matM, sizeof(NODAL_PRECISION) * g.nxyz() * g.ng2()));
	checkCudaErrors(cudaMalloc((void**)&_matMI, sizeof(NODAL_PRECISION) * g.nxyz() * g.ng2()));
	checkCudaErrors(cudaMalloc((void**)&_matMs, sizeof(NODAL_PRECISION) * g.nxyz() * g.ng2()));
	checkCudaErrors(cudaMalloc((void**)&_matMf, sizeof(NODAL_PRECISION) * g.nxyz() * g.ng2()));
}

NodalCuda::~NodalCuda()
{
}

void NodalCuda::init()
{
}

void NodalCuda::reset(CrossSection& xs, double& reigv, NODAL_PRECISION* jnet, double* flux)
{

	_reigv = reigv;
	checkCudaErrors(cudaMemcpy(_flux, flux, sizeof(double) * _g.nxyz() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_jnet, jnet, sizeof(NODAL_PRECISION) * _g.nsurf() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsnf, &xs.xsnf(0, 0), sizeof(XS_PRECISION) * _g.nxyz() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsdf, &xs.xsdf(0, 0), sizeof(XS_PRECISION) * _g.nxyz() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xstf, &xs.xstf(0, 0), sizeof(XS_PRECISION) * _g.nxyz() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_chif, &xs.chif(0, 0), sizeof(XS_PRECISION) * _g.nxyz() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsadf, &xs.xsadf(0, 0), sizeof(XS_PRECISION) * _g.nxyz() * _g.ng(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xssf, &xs.xssf(0, 0, 0), sizeof(XS_PRECISION) * _g.nxyz() * _g.ng2(), cudaMemcpyHostToDevice));
}

void NodalCuda::drive(NODAL_PRECISION* jnet)
{
#ifdef _DEBUG
	NODAL_PRECISION* temp = new NODAL_PRECISION[_g.nxyz() * NDIRMAX * _g.ng2()]{};
#endif
	::updateConstant << <_blocks, _threads >> > (*this);
	checkCudaErrors(cudaDeviceSynchronize());
#ifdef _DEBUG
	checkCudaErrors(cudaMemcpy(temp, _m264, sizeof(NODAL_PRECISION) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _diagDI, sizeof(NODAL_PRECISION) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
#endif

	::caltrlcff0 << <_blocks, _threads>> > (*this);
	cudaDeviceSynchronize();
	::caltrlcff12 << <_blocks, _threads >> > (*this);
	cudaDeviceSynchronize();
#ifdef _DEBUG
	checkCudaErrors(cudaMemcpy(temp, _trlcff0, sizeof(NODAL_PRECISION) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _trlcff1, sizeof(NODAL_PRECISION) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _trlcff2, sizeof(NODAL_PRECISION) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
#endif

	::updateMatrix << <_blocks, _threads >> > (*this);
	cudaDeviceSynchronize();

#ifdef _DEBUG
	checkCudaErrors(cudaMemcpy(temp, _mu, sizeof(NODAL_PRECISION) * _g.nxyz() * NDIRMAX * _g.ng2(), cudaMemcpyDeviceToHost));
#endif


	::calculateEven << <_blocks, _threads >> > (*this);
	cudaDeviceSynchronize();

#ifdef _DEBUG
	checkCudaErrors(cudaMemcpy(temp, _dsncff4, sizeof(NODAL_PRECISION) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _dsncff6, sizeof(NODAL_PRECISION) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _dsncff2, sizeof(NODAL_PRECISION) * _g.nxyz() * NDIRMAX * _g.ng(), cudaMemcpyDeviceToHost));
#endif


	::calculateJnet << <_blocks_sfc, _threads_sfc >> > (*this);
	cudaDeviceSynchronize();

	checkCudaErrors(cudaMemcpy(jnet, _jnet, sizeof(NODAL_PRECISION) * _g.nsurf() * _g.ng(), cudaMemcpyDeviceToHost));

#ifdef _DEBUG
	delete[] temp;
#endif
}
