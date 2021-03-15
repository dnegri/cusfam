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

class Nodal : public Managed {
protected:
    Geometry& _g;

    int* _ng;
    int* _ng2;
    int* _nxyz;
    int* _nsurf;
    int* _symopt;
    int* _symang;

	GEOM_VAR* _albedo;

    int* _neib;
    int* _lktosfc;
	GEOM_VAR* _hmesh;

    int* _lklr;
    int* _idirlr;
    int* _sgnlr;

    XS_VAR* _xstf;
    XS_VAR* _xsdf;
    XS_VAR* _xsnf;
    XS_VAR* _chif;
    XS_VAR* _xssf;
    XS_VAR* _xsadf;

    NODAL_VAR* _trlcff0;
    NODAL_VAR* _trlcff1;
    NODAL_VAR* _trlcff2;
    NODAL_VAR* _eta1;
    NODAL_VAR* _eta2;
    NODAL_VAR* _mu;
    NODAL_VAR* _tau;


    NODAL_VAR* _m260;
    NODAL_VAR* _m251;
    NODAL_VAR* _m253;
    NODAL_VAR* _m262;
    NODAL_VAR* _m264;

    NODAL_VAR* _diagDI;
    NODAL_VAR* _diagD;
    NODAL_VAR* _matM;
    NODAL_VAR* _matMI;
    NODAL_VAR* _matMs;
    NODAL_VAR* _matMf;

    NODAL_VAR* _dsncff2;
    NODAL_VAR* _dsncff4;
    NODAL_VAR* _dsncff6;

    SOL_VAR* _jnet;
    SOL_VAR* _flux;
    double _reigv;
public:
	int nmaxswp;
	int nlupd;
	int nlswp;

public:
    __host__ Nodal(Geometry& g);
    __host__  virtual ~Nodal();

    __host__ virtual void init() =0 ;
    __host__ virtual void reset(CrossSection& xs, const double& reigv, SOL_VAR* jnet, SOL_VAR* phif) = 0;
    __host__ virtual void drive(SOL_VAR* jnet) = 0;


	__host__ __device__ void updateConstant(const int& lk);
    __host__ __device__ void updateMatrix(const int& lk);
    __host__ __device__ void trlcffbyintg(NODAL_VAR* avgtrl3, NODAL_VAR* hmesh3, NODAL_VAR& trlcff1, NODAL_VAR& trlcff2);
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