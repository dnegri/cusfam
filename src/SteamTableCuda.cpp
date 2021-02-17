#include "SteamTableCuda.h"


SteamTableCuda::SteamTableCuda(SteamTable& steam)
{
	_steam_cpu = &steam;

	checkCudaErrors(cudaMalloc((void**)&_press, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_tmin, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_tmax, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_dmin, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_dmax, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_dgas, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_hmin, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_hmax, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_hgas, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_vismin, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_vismax, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_visgas, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_tcmin, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_tcmax, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_tcgas, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_shmin, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_shmax, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_shgas, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_rhdel, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_rtdel, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&_rhdiff, sizeof(float)));


	checkCudaErrors(cudaMalloc((void**)&_cmn, sizeof(float) * 8));
	checkCudaErrors(cudaMalloc((void**)&_hmod, sizeof(float) * _np * _npnts));
	checkCudaErrors(cudaMalloc((void**)&_dmodref, sizeof(float) * _np * _npnts));
	checkCudaErrors(cudaMalloc((void**)&_propc, sizeof(float) * _nprop * _np * _npnts));

	checkCudaErrors(cudaMemcpy(_press, &_steam_cpu->press(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_tmin, &_steam_cpu->tmin(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_tmax, &_steam_cpu->tmax(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmin, &_steam_cpu->dmin(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmax, &_steam_cpu->dmax(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dgas, &_steam_cpu->dgas(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hmin, &_steam_cpu->hmin(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hmax, &_steam_cpu->hmax(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hgas, &_steam_cpu->hgas(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_vismin, &_steam_cpu->vismin(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_vismax, &_steam_cpu->vismax(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_visgas, &_steam_cpu->visgas(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_tcmin, &_steam_cpu->tcmin(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_tcmax, &_steam_cpu->tcmax(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_tcgas, &_steam_cpu->tcgas(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_shmin, &_steam_cpu->shmin(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_shmax, &_steam_cpu->shmax(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_shgas, &_steam_cpu->shgas(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_rhdel, &_steam_cpu->rhdel(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_rtdel, &_steam_cpu->rtdel(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_rhdiff, &_steam_cpu->rhdiff(), sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(_cmn, _steam_cpu->cmn(), sizeof(float) * 8, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hmod, _steam_cpu->hmod(), sizeof(float) * _np * _npnts, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmodref, _steam_cpu->dmodref(), sizeof(float) * _np * _npnts, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_propc, _steam_cpu->propc(), sizeof(float) * _nprop * _np * _npnts, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());
}

SteamTableCuda::~SteamTableCuda()
{
}

void SteamTableCuda::setPressure(const float& press)
{
	if (abs(_steam_cpu->getPressure() - press) < 0.01) return;

	_steam_cpu->setPressure(press);

	checkCudaErrors(cudaMemcpy(_press, &_steam_cpu->press(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_tmin, &_steam_cpu->tmin(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_tmax, &_steam_cpu->tmax(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmin, &_steam_cpu->dmin(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmax, &_steam_cpu->dmax(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dgas, &_steam_cpu->dgas(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hmin, &_steam_cpu->hmin(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hmax, &_steam_cpu->hmax(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hgas, &_steam_cpu->hgas(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_vismin, &_steam_cpu->vismin(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_vismax, &_steam_cpu->vismax(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_visgas, &_steam_cpu->visgas(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_tcmin, &_steam_cpu->tcmin(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_tcmax, &_steam_cpu->tcmax(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_tcgas, &_steam_cpu->tcgas(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_shmin, &_steam_cpu->shmin(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_shmax, &_steam_cpu->shmax(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_shgas, &_steam_cpu->shgas(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_rhdel, &_steam_cpu->rhdel(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_rtdel, &_steam_cpu->rtdel(), sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_rhdiff, &_steam_cpu->rhdiff(), sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(_cmn, _steam_cpu->cmn(), sizeof(float) * 8, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hmod, _steam_cpu->hmod(), sizeof(float) * _np * _npnts, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmodref, _steam_cpu->dmodref(), sizeof(float) * _np * _npnts, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_propc, _steam_cpu->propc(), sizeof(float) * _nprop * _np * _npnts, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());
}
