#pragma once
#include "pch.h"
#include "Depletion.h"
#include "GeometryCuda.h"

class DepletionCuda : public Depletion
{
public:
	__host__ DepletionCuda(GeometryCuda& g);
	__host__ virtual ~DepletionCuda();
	__host__  void init();
	__host__  void dep(const float& tsec);
	__host__  void eqxe(const float* xsmica, const float* xsmicf, const double* flux, const float& fnorm);
	__host__  void pickData(const float* xsmica, const float* xsmicf, const float* xsmic2n, const double* flux, const float& fnorm);
	__host__  void updateH2ODensity(const float* dm, const float& ppm);

	__host__  void setDensity(const float* dnst);
	__host__  void setBurnup(const float* burn);
	__host__  void setH2ORatio(const float* h2on);
};

