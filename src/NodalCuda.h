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

	int* _neib;
	int* _lktosfc;
	float* _hmesh;

	int* _lklr;
	int* _idirlr;
	int* _sgnlr;

	XS_PRECISION* _xstf;
	XS_PRECISION* _xsdf;
	XS_PRECISION* _xsnff;
	XS_PRECISION* _xschif;
	XS_PRECISION* _xssf;
	XS_PRECISION* _xsadf;


	float* _trlcff0;
	float* _trlcff1;
	float* _trlcff2;
	double* _flux;
	float* _jnet;
	float* _eta1;
	float* _eta2;
	float* _mu;
	float* _tau;


	float* _m260;
	float* _m251;
	float* _m253;
	float* _m262;
	float* _m264;

	float* _diagDI;
	float* _diagD;
	float* _matM;
	float* _matMI;
	float* _matMs;
	float* _matMf;

	float* _dsncff2;
	float* _dsncff4;
	float* _dsncff6;

	double _reigv;

	dim3 _blocks, _blocks_sfc;
	dim3 _threads, _threads_sfc;

public:
	NodalCuda(Geometry& g);
	virtual ~NodalCuda();

	void init();
	void reset(CrossSection& xs);
	void drive();

	inline float& trlcff0(const int& ig, const int& idir, const int& lk) { return _trlcff0[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& trlcff1(const int& ig, const int& idir, const int& lk) { return _trlcff1[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& trlcff2(const int& ig, const int& idir, const int& lk) { return _trlcff2[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& jnet(const int& ig, const int& idir, const int& lk) { return _jnet[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& eta1(const int& ig, const int& idir, const int& lk) { return _eta1[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& eta2(const int& ig, const int& idir, const int& lk) { return _eta2[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& mu(const int& ig, const int& idir, const int& lk) { return _mu[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& tau(const int& ig, const int& idir, const int& lk) { return _tau[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& m260(const int& ig, const int& idir, const int& lk) { return _m260[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& m251(const int& ig, const int& idir, const int& lk) { return _m251[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& m253(const int& ig, const int& idir, const int& lk) { return _m253[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& m262(const int& ig, const int& idir, const int& lk) { return _m262[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& m264(const int& ig, const int& idir, const int& lk) { return _m264[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& diagDI(const int& ig, const int& idir, const int& lk) { return _diagDI[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& diagD(const int& ig, const int& idir, const int& lk) { return _diagD[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& dsncff2(const int& ig, const int& idir, const int& lk) { return _dsncff2[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& dsncff4(const int& ig, const int& idir, const int& lk) { return _dsncff4[(lk * NDIRMAX + idir) * _ng + ig]; };
	inline float& dsncff6(const int& ig, const int& idir, const int& lk) { return _dsncff6[(lk * NDIRMAX + idir) * _ng + ig]; };

	inline XS_PRECISION& xstf(const int& ig, const int& lk) { return _xstf[lk * _ng + ig]; };
	inline XS_PRECISION& xsdf(const int& ig, const int& lk) { return _xsdf[lk * _ng + ig]; };
	inline XS_PRECISION& xsnff(const int& ig, const int& lk) { return _xsnff[lk * _ng + ig]; };
	inline XS_PRECISION& xschif(const int& ig, const int& lk) { return _xschif[lk * _ng + ig]; };
	inline XS_PRECISION& xsadf(const int& ig, const int& lk) { return _xsadf[lk * _ng + ig]; };

	inline XS_PRECISION& xssf(const int& igs, const int& igd, const int& lk) { return _xssf[lk * _ng2 + igd * _ng + igs]; };
	inline float& matM(const int& igs, const int& igd, const int& lk) { return _matM[lk * _ng2 + igd * _ng + igs]; };
	inline float& matMI(const int& igs, const int& igd, const int& lk) { return _matMI[lk * _ng2 + igd * _ng + igs]; };
	inline float& matMs(const int& igs, const int& igd, const int& lk) { return _matMs[lk * _ng2 + igd * _ng + igs]; };
	inline float& matMf(const int& igs, const int& igd, const int& lk) { return _matMf[lk * _ng2 + igd * _ng + igs]; };

	inline double& flux(const int& ig, const int& lk) { return _flux[lk * _ng + ig]; };

	inline double& reigv() { return _reigv; };


};

