#pragma once

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
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
#include "JIArray.h"
#include "ShapeMatch.h"

namespace py = pybind11;

enum ShapeMatchOption {
    SHAPE_NO,
    SHAPE_HOLD,
    SHAPE_MATCH
};

enum CriticalOption {
    KEFF,
    CBC
};
enum DepletionIsotope {
    DEP_ALL,
    DEP_FP,
    DEP_XE
};

typedef struct _SimonGeometry {
	int nz;
	int nxya;
	int kbc;
	int kec;
	float core_height;
	py::list hz;
} SimonGeometry;




typedef struct _SimonResult {
	_SimonResult(int nxya, int nz) : pow2d(nxya), pow1d(nz) {}
	int    error;
	double eigv;
	float ppm;
	float fq;
	float fxy;
	float fr;
	float fz;
	float asi;
	float tf;
	float tm;
	py::dict rod_pos;
	py::list pow2d;
	py::list pow1d;
} SimonResult;



typedef struct _SteadyOption {
    CriticalOption   searchOption = CriticalOption::KEFF;
    ShapeMatchOption shpmtch = ShapeMatchOption::SHAPE_NO;
    bool feedtf = true;
    bool feedtm = true;
    XEType xenon = XEType::XE_EQ;
	SMType samarium = SMType::SM_TR;
	float tin = 290.0;
    double eigvt = 1.0;
    int maxiter = 100;
	float epsiter = 1.E-5;
	float ppm = 500.0;
	float plevel = 1.0;
    float b10a = 0.0;
} SteadyOption ;

typedef struct _DepletionOption {
	DepletionIsotope isotope;
	XEType xe;
	SMType sm;
	float  tsec;
	float  xeamp = 1.0;
} DepletionOption;

class Simon : public Managed {
protected:
    Geometry* _g;
    SteamTable* _steam;
    CrossSection* _x;
    Depletion* _d;
    Feedback* _f;
	ControlRod* _r;
    ShapeMatch* _shapeMatch;

	void* _tset_ptr;
	void* _ff_ptr;

    int _nstep;
    float _epsbu;

    float* _bucyc;
    float* _power;
	float* _pow1d;
	float* _pow2d;
	float* _pow2da;
	float* _pow3da;

	float _pload;
	float _pload0;

    double* _flux;
    double* _jnet;
	double* _phis;
    double1 _powshp;

    double _reigv;
    double _eigv;
    double _eigvc;
    double _fnorm;

    float _ppm;
    float _press;
    float _tin;
	float _asi;
	float _fxy;
	float _fq;
	float _fr;

    bool _feed_tf;
    bool _feed_tm;

	//FIXME volcore should be defined in Geometry
	double _volcore;


public:

    __host__ Simon();
    __host__ virtual ~Simon();

    __host__ __device__ Geometry& g() { return *_g; };
    __host__ __device__ CrossSection& x() { return *_x; };
    __host__ __device__ Feedback& f() { return *_f; };
    __host__ __device__ Depletion& d() { return *_d; };
    __host__ __device__ SteamTable& steam() { return *_steam; };
	__host__ __device__ ControlRod& r() { return *_r; };

	__host__ void setRodPosition(const char* rodid, const float& position);

	__host__ virtual void setBurnup(const char* dir_burn, const float& burnup, SteadyOption& option);
	__host__ virtual void setSMR(const char* dir_burn, const float& burnup);
	__host__ virtual void saveSMR(const char* dir_burn, const float& burnup);
	__host__ virtual void setBurnupPoints(const std::vector<double> & burnups);
	__host__ virtual void initialize(const char* dbfile);
	__host__ virtual void readTableSet(const char* tsetfile);
	__host__ virtual void readFormFunction(const char* fffile);
	__host__ virtual void updateBurnup();

    __host__ virtual void runSteady(const SteadyOption& condition) = 0;
    __host__ virtual void runKeff(const int& nmaxout)=0;
    __host__ virtual void runECP(const int& nmaxout, const double& eigvt) =0;
    __host__ virtual void runDepletion(const DepletionOption& option) = 0;
    __host__ virtual void runXenonTransient(const DepletionOption& option) = 0;
    __host__ virtual void normalize() = 0;

    __host__ inline float* power() {return _power;};
	__host__ inline float& power(const int& l) { return _power[l]; };
	__host__ inline float& power(const int& l2d, const int& k) { return _power[k*_g->nxy()+l2d]; };
	__host__ inline float* pow3da() { return _pow3da; };
	__host__ inline float& pow3da(const int& lka) { return _pow3da[lka]; };
	__host__ inline float& pow3da(const int& l2da, const int& k) { return _pow3da[k * _g->nxya() + l2da]; };
	__host__ inline float* pow1d() { return _pow1d; };
	__host__ inline float* pow2d() { return _pow2d; };
	__host__ inline float* pow2da() { return _pow2da; };
	__host__ inline float& pow1d(const int& k) { return _pow1d[k]; };
	__host__ inline float& pow2d(const int& l2d) { return _pow2d[l2d]; };
	__host__ inline float& pow2da(const int& l2da) { return _pow2da[l2da]; };
	__host__ inline double& flux(const int& ig, const int& l) { return _flux[l*_g->ng()+ig]; };
    __host__ inline double& jnet(const int& ig, const int& ls) { return _jnet[ls*_g->ng()+ig]; };
    __host__ inline double* flux() { return _flux; };
    __host__ inline double* jnet() { return _jnet; };
	__host__ inline float& asi() { return _asi; };
	__host__ inline float& ppm() { return _ppm; };
	__host__ inline double& eigv() { return _eigv; };
	__host__ inline double& fnorm() { return _fnorm; };
	__host__ inline float fxy() { return _fxy; };
	__host__ inline float fr() { return _fr; };
	__host__ inline float fq() { return _fq; };
	__host__ inline float& pload() { return _pload; };
	__host__ inline const int& nburn() { return _nstep; };

	__host__ inline float& burn(const int& istep) { return _bucyc[istep]; };
	__host__ inline float dburn(const int& istep) { 
		return _bucyc[istep] - _bucyc[istep-1];
	};

    __host__ void print(Geometry& g, CrossSection& x, Feedback& f, Depletion& d);


	void generateResults();

    void setPowerShape(const vector<double>& hzshp, const vector<double>& powshp);

};


