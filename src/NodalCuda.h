#pragma once
#include "Nodal.h"
#include "CrossSection.h"

class NodalCuda : public Nodal
{
private:

	dim3 _blocks, _blocks_sfc;
	dim3 _threads, _threads_sfc;

public:
	__host__ NodalCuda(Geometry& g);
	__host__ virtual ~NodalCuda();

	__host__ void init();
	__host__ void reset(CrossSection& xs, double& reigv, NODAL_PRECISION* jnet, double* phif);
	__host__ void drive(NODAL_PRECISION* jnet);
};

