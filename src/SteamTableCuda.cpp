#include "SteamTableCuda.h"


SteamTableCuda::SteamTableCuda(const SteamTable& steam)
{

	_press = steam.press();
	_tmin = steam.tmin();
	_tmax = steam.tmax();
	_dmin = steam.dmin();
	_dmax = steam.dmax();
	_dgas = steam.dgas();
	_hmin = steam.hmin();
	_hmax = steam.hmax();
	_hgas = steam.hgas();
	_vismin = steam.vismin();
	_vismax = steam.vismax();
	_visgas = steam.visgas();
	_tcmin = steam.tcmin();
	_tcmax = steam.tcmax();
	_tcgas = steam.tcgas();
	_shmin = steam.shmin();
	_shmax = steam.shmax();
	_shgas = steam.shgas();
	_rhdel = steam.rhdel();
	_rtdel = steam.rtdel();
	_rhdiff = steam.rhdiff();

	checkCudaErrors(cudaMalloc((void**)&_cmn, sizeof(float) * 8));
	checkCudaErrors(cudaMalloc((void**)&_hmod, sizeof(float) * _np * _npnts));
	checkCudaErrors(cudaMalloc((void**)&_dmodref, sizeof(float) * _np * _npnts));
	checkCudaErrors(cudaMalloc((void**)&_propc, sizeof(float) * _nprop * _np * _npnts));

	checkCudaErrors(cudaMemcpy(_cmn, steam.cmn(), sizeof(float) * 8, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hmod, steam.hmod(), sizeof(float) * _np * _npnts, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmodref, steam.dmodref(), sizeof(float) * _np * _npnts, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_propc, steam.propc(), sizeof(float) * _nprop * _np * _npnts, cudaMemcpyHostToDevice));
}

SteamTableCuda::~SteamTableCuda()
{
}
