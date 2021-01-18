#pragma once
#include "pch.h"
#include "Geometry.h"
#include "CrossSection.h"


#define m011   0.666666667
#define m022   0.4
#define m033   0.285714286
#define m044   0.222222222
#define m220   6.
#define rm220  0.166666667
#define m240   20.
#define m231   10.
#define m242   14.

class Nodal {
protected:
    Geometry& _g;

    int* _ng;
    int* _ng2;
    int* _nxyz;
    int* _nsurf;
    int* _symopt;
    int* _symang;

    float* _albedo;

    int* _neib;
    int* _lktosfc;
    float* _hmesh;

    int* _lklr;
    int* _idirlr;
    int* _sgnlr;

    XS_PRECISION* _xstf;
    XS_PRECISION* _xsdf;
    XS_PRECISION* _xsnf;
    XS_PRECISION* _chif;
    XS_PRECISION* _xssf;
    XS_PRECISION* _xsadf;

    float* _trlcff0;
    float* _trlcff1;
    float* _trlcff2;
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

    float* _jnet;
    double* _flux;
    double _reigv;
public:
	int nmaxswp;
	int nlupd;
	int nlswp;

public:
    __device__ __host__ Nodal(Geometry& g);
    __device__ __host__  virtual ~Nodal();

	__host__ __device__ void updateConstant(const int& lk);
    __host__ __device__ void updateMatrix(const int& lk);
    __host__ __device__ void trlcffbyintg(float* avgtrl3, float* hmesh3, float& trlcff1, float& trlcff2);
    __host__ __device__ void calculateTransverseLeakage(const int& lk);
    __host__ __device__ void calculateEven(const int& lk);
    __host__ __device__ void calculateJnet(const int& ls);
    __host__ __device__ void calculateJnet1n(const int& ls, const int& lr, const float& alb);
    __host__ __device__ void calculateJnet2n(const int& ls);

    __host__ __device__ inline int& ng() { return *_ng; };
    __host__ __device__ inline int& ng2() { return *_ng2; };
    __host__ __device__ inline int& nxyz() { return *_nxyz; };
    __host__ __device__ inline int& nsurf() { return *_nsurf; };

    __host__ __device__ inline int* nxyz1() { return _nxyz; };

};