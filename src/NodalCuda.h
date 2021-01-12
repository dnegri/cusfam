#pragma once
#include <cuda_runtime.h>
#include "Nodal.h"
#include "CrossSection.h"

class NodalCuda : public Nodal
{
private:
	int _ng;
	int _ng2;
	int _nxyz;
	int _nsurf;

	int* _d_nxyz;
	int* _d_nsurf;

	int* _d_symopt;
	int* _d_symang;
	float* _d_albedo;

	int* _d_neib;
	int* _d_lktosfc;
	float* _d_hmesh;

	int* _d_lklr;
	int* _d_idirlr;
	int* _d_sgnlr;

	XS_PRECISION* _d_xstf;
	XS_PRECISION* _d_xsdf;
	XS_PRECISION* _d_xsnff;
	XS_PRECISION* _d_xschif;
	XS_PRECISION* _d_xssf;
	XS_PRECISION* _d_xsadf;

	float*	_d_jnet;
	double* _d_flux;
	double* _d_reigv;

	float* _d_trlcff0;
	float* _d_trlcff1;
	float* _d_trlcff2;
	float* _d_eta1;
	float* _d_eta2;
	float* _d_mu;
	float* _d_tau;


	float* _d_m260;
	float* _d_m251;
	float* _d_m253;
	float* _d_m262;
	float* _d_m264;

	float* _d_diagDI;
	float* _d_diagD;
	float* _d_matM;
	float* _d_matMI;
	float* _d_matMs;
	float* _d_matMf;

	float* _d_dsncff2;
	float* _d_dsncff4;
	float* _d_dsncff6;

	float* _jnet;
	double* _flux;
	double _reigv;

	dim3 _d_blocks, _d_blocks_sfc;
	dim3 _d_threads, _d_threads_sfc;

public:
	NodalCuda(Geometry& g);
	virtual ~NodalCuda();

	void init();
	void reset(CrossSection& xs, double* reigv, double* jnet, double* phif);
	void drive();

	inline float& jnet(const int& ig, const int& lks) { return _jnet[lks * _ng + ig]; };
	inline double& flux(const int& ig, const int& lk) { return _flux[lk * _ng + ig]; };

	inline double& reigv() { return _reigv; };


};

