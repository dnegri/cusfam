#pragma once

#include "pch.h"
#include "CrossSection.h"
#include "Geometry.h"
#include "SteamTable.h"
#include "SteamTable.h"
#include "NodalCPU.h"
#include "CMFDCPU.h"
#include "BICGCMFD.h"
#include "Geometry.h"
#include "Depletion.h"
#include "CrossSection.h"
#include "Feedback.h"

enum CriticalOption {
    KEFF,
    CBC
};
enum DepletionIsotope {
    DEP_ALL,
    DEP_FP,
    DEP_XE
};

typedef struct _SteadyOption {
    CriticalOption   searchOption;
    bool feedtf;
    bool feedtm;
    XEType xenon;
    float tin;
    double eigvt;
    int maxiter;
	float ppm;
	float plevel;
} SteadyOption ;

typedef struct _DepletionOption {
	DepletionIsotope isotope;
	XEType xe;
	SMType sm;
	float  tsec;
} DepletionOption;

class Simon : public Managed {
protected:
    Geometry* _g;
    SteamTable* _steam;
    CrossSection* _x;
    Depletion* _d;
    Feedback* _f;

    int _nstep;
    float _epsbu;

    float* _bucyc;
    float* _power;

	float _pload;
	float _pload0;

    SOL_VAR* _flux;
    SOL_VAR* _jnet;
    double _reigv;
    double _eigv;
    double _fnorm;

    float _ppm;
    float _press;
    float _tin;

    bool _feed_tf;
    bool _feed_tm;

public:

    __host__ Simon();
    __host__ virtual ~Simon();

    __host__ __device__ Geometry& g() { return *_g; };
    __host__ __device__ CrossSection& x() { return *_x; };
    __host__ __device__ Feedback& f() { return *_f; };
    __host__ __device__ Depletion& d() { return *_d; };
    __host__ __device__ SteamTable& steam() { return *_steam; };

    __host__ void setBurnup(const float& burnup);

    __host__ virtual void initialize(const char* dbfile);
    __host__ virtual void runSteady(const SteadyOption& condition) = 0;
    __host__ virtual void runKeff(const int& nmaxout)=0;
    __host__ virtual void runECP(const int& nmaxout, const double& eigvt) =0;
    __host__ virtual void runDepletion(const DepletionOption& option) = 0;
    __host__ virtual void runXenonTransient() = 0;
    __host__ virtual void normalize() = 0;

    __host__ inline float* power() {return _power;};
    __host__ inline float& power(const int& l) { return _power[l]; };
    __host__ inline SOL_VAR& flux(const int& ig, const int& l) { return _flux[l*_g->ng()+ig]; };
    __host__ inline SOL_VAR& jnet(const int& ig, const int& ls) { return _jnet[ls*_g->ng()+ig]; };
    __host__ inline SOL_VAR* flux() { return _flux; };
    __host__ inline SOL_VAR* jnet() { return _jnet; };
    __host__ inline float& ppm() { return _ppm; };
	__host__ inline double& eigv() { return _eigv; };
	__host__ inline double& fnorm() { return _fnorm; };
	__host__ inline float& pload() { return _pload; };
	__host__ inline const int& nburn() { return _nstep; };

	__host__ inline float& burn(const int& istep) { return _bucyc[istep]; };
	__host__ inline float dburn(const int& istep) { 
		return _bucyc[istep+1] - _bucyc[istep];
	};

    __host__ void print(Geometry& g, CrossSection& x, Feedback& f, Depletion& d);




};


