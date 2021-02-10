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
    double* _flux;
    double* _jnet;
    double _reigv;
    double _eigv;
    double _pload;
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

    __host__ void initialize(const char* dbfile);
    __host__ void setBurnup(const float& burnup);
    __host__ void setFeedbackOption(bool feed_tf, bool feed_tm);

    __host__ virtual void runKeff(const int& nmaxout)=0;
    __host__ virtual void runECP(const int& nmaxout, const double& eigvt) =0;
    __host__ virtual void runDepletion(const float& dburn) = 0;
    __host__ virtual void runXenonTransient() = 0;
    __host__ virtual void normalize() = 0;

    __host__ inline float* power() {return _power;};
    __host__ inline float& power(const int& l) { return _power[l]; };
    __host__ inline double& flux(const int& ig, const int& l) { return _flux[l*_g->ng()+ig]; };




};


