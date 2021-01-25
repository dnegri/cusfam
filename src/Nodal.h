﻿#pragma once
#include "pch.h"
#include "Geometry.h"
#include "CrossSection.h"

#define NODAL_PRECISION double

#define m011   0.666666667
#define m022   0.4
#define m033   0.285714286
#define m044   0.222222222
#define m220   6.
#define rm220  0.166666667
#define m240   20.
#define m231   10.
#define m242   14.

class Nodal : public Managed {
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

    NODAL_PRECISION* _trlcff0;
    NODAL_PRECISION* _trlcff1;
    NODAL_PRECISION* _trlcff2;
    NODAL_PRECISION* _eta1;
    NODAL_PRECISION* _eta2;
    NODAL_PRECISION* _mu;
    NODAL_PRECISION* _tau;


    NODAL_PRECISION* _m260;
    NODAL_PRECISION* _m251;
    NODAL_PRECISION* _m253;
    NODAL_PRECISION* _m262;
    NODAL_PRECISION* _m264;

    NODAL_PRECISION* _diagDI;
    NODAL_PRECISION* _diagD;
    NODAL_PRECISION* _matM;
    NODAL_PRECISION* _matMI;
    NODAL_PRECISION* _matMs;
    NODAL_PRECISION* _matMf;

    NODAL_PRECISION* _dsncff2;
    NODAL_PRECISION* _dsncff4;
    NODAL_PRECISION* _dsncff6;

    NODAL_PRECISION* _jnet;
    double* _flux;
    double _reigv;
public:
	int nmaxswp;
	int nlupd;
	int nlswp;

public:
    __host__ Nodal(Geometry& g);
    __host__  virtual ~Nodal();

    __host__ virtual void init() =0 ;
    __host__ virtual void reset(CrossSection& xs, double& reigv, NODAL_PRECISION* jnet, double* phif) = 0;
    __host__ virtual void drive(NODAL_PRECISION* jnet) = 0;


	__host__ __device__ void updateConstant(const int& lk);
    __host__ __device__ void updateMatrix(const int& lk);
    __host__ __device__ void trlcffbyintg(NODAL_PRECISION* avgtrl3, NODAL_PRECISION* hmesh3, NODAL_PRECISION& trlcff1, NODAL_PRECISION& trlcff2);
    __host__ __device__ void caltrlcff0(const int& lk);
    __host__ __device__ void caltrlcff12(const int& lk);
    __host__ __device__ void calculateEven(const int& lk);
    __host__ __device__ void calculateJnet(const int& ls);
    __host__ __device__ void calculateJnet1n(const int& ls, const int& lr, const float& alb);
    __host__ __device__ void calculateJnet2n(const int& ls);

    __host__ __device__ inline int& ng() { return *_ng; };
    __host__ __device__ inline int& ng2() { return *_ng2; };
    __host__ __device__ inline int& nxyz() { return *_nxyz; };
    __host__ __device__ inline int& nsurf() { return *_nsurf; };

};