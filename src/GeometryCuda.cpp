#include "GeometryCuda.h"

GeometryCuda::GeometryCuda(const Geometry& g)
{

	_ng = g.ng();
	_ng2 = g.ng2();
	_nxy = g.nxy();
	_nz = g.nz();
	_nxyz = g.nxyz();
	_ngxy = g.ngxy();
	_ngxyz = g.ngxyz();
	_nx = g.nx();
	_ny = g.ny();
	_nsurf = g.nsurf();
	_symopt = g.symopt();
	_symang = g.symang();

	checkCudaErrors(cudaMalloc((void**)&_albedo, sizeof(float) * LR * NDIRMAX));
	checkCudaErrors(cudaMalloc((void**)&_neibr, sizeof(int) * _nxy * NEWS));
	checkCudaErrors(cudaMalloc((void**)&_ijtol, sizeof(int) * _nx * _ny));
	checkCudaErrors(cudaMalloc((void**)&_nxs, sizeof(int) * _ny));
	checkCudaErrors(cudaMalloc((void**)&_nxe, sizeof(int) * _ny));
	checkCudaErrors(cudaMalloc((void**)&_nys, sizeof(int) * _nx));
	checkCudaErrors(cudaMalloc((void**)&_nye, sizeof(int) * _nx));
	checkCudaErrors(cudaMalloc((void**)&_neib, sizeof(int) * _nxyz * LR * NDIRMAX));
	checkCudaErrors(cudaMalloc((void**)&_hmesh, sizeof(float) * _nxyz * LR * NDIRMAX));
	checkCudaErrors(cudaMalloc((void**)&_lktosfc, sizeof(int) * _nxyz * LR * NDIRMAX));
	checkCudaErrors(cudaMalloc((void**)&_vol, sizeof(float) * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_idirlr, sizeof(int) * _nsurf * LR));
	checkCudaErrors(cudaMalloc((void**)&_sgnlr, sizeof(int) * _nsurf * LR));
	checkCudaErrors(cudaMalloc((void**)&_lklr, sizeof(int) * _nsurf * LR));

	checkCudaErrors(cudaMemcpy(_albedo, g.albedo(), sizeof(float) * LR * NDIRMAX, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_neibr, g.neibr(), sizeof(int) * _nxy * NEWS, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_ijtol, g.ijtol(), sizeof(int) * _nx * _ny, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_nxs, g.nxs(), sizeof(int) * _ny, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_nxe, g.nxe(), sizeof(int) * _ny, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_nys, g.nys(), sizeof(int) * _nx, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_nye, g.nye(), sizeof(int) * _nx, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_neib, g.neib(), sizeof(int) * _nxyz * LR * NDIRMAX, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hmesh, g.hmesh(), sizeof(float) * _nxyz * LR * NDIRMAX, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_lktosfc, g.lktosfc(), sizeof(int) * _nxyz * LR * NDIRMAX, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_vol, g.vol(), sizeof(float) * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_idirlr, g.idirlr(), sizeof(int) * _nsurf * LR, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_sgnlr, g.sgnlr(), sizeof(int) * _nsurf * LR, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_lklr, g.lklr(), sizeof(int) * _nsurf * LR, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
}

GeometryCuda::~GeometryCuda()
{
}
