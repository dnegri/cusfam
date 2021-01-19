#pragma once
#include "Nodal.h"
#include "CrossSection.h"

class NodalCuda : public Nodal
{
private:
	float* _host_jnet;
	double* _host_flux;
	double _host_reigv;

	dim3 _blocks, _blocks_sfc;
	dim3 _threads, _threads_sfc;

public:
	__host__ NodalCuda(Geometry& g);
	__host__ virtual ~NodalCuda();

	__host__ void init();
	__host__ void reset(CrossSection& xs, double* reigv, double* jnet, double* phif);
	__host__ void drive();

	inline float& host_jnet(const int& ig, const int& lks) { return _host_jnet[lks * _g.ng() + ig]; };
	inline double& host_flux(const int& ig, const int& lk) { return _host_flux[lk * _g.ng() + ig]; };

	inline double& host_reigv() { return _host_reigv; };


};

