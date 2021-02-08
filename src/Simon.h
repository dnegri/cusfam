#pragma once

#include "pch.h"
#include "CrossSection.h"
#include "Geometry.h"
#include "SteamTable.h"
#include "NodalCPU.h"
#include "CMFDCPU.h"
#include "BICGCMFD.h"
#include "Geometry.h"
#include "DepletionChain.h"
#include "CrossSection.h"
#include "Feedback.h"


class Simon {
private:
    Geometry* _g;
    SteamTable* _steam;
    CrossSection* _x;
    DepletionChain* _d;
    Feedback* _f;

    BICGCMFD* cmfd;

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

    Simon();
    virtual ~Simon();

    Geometry& g() { return *_g; };
    CrossSection& x() { return *_x; };
    Feedback& f() { return *_f; };
    DepletionChain& d() { return *_d; };
    SteamTable& steam() { return *_steam; };

    void initialize(const char* dbfile);
    void setBurnup(const float& burnup);
    void setFeedbackOption(bool feed_tf, bool feed_tm);

    void runKeff(const int& nmaxout);
    void runECP(const int& nmaxout, const double& eigvt);
    void runDepletion(const float& dburn);
    void runXenonTransient();

    inline float* power() {return _power;};
    inline float& power(const int& l) { return _power[l]; };
    inline double& flux(const int& ig, const int& l) { return _flux[l*_g->ng()+ig]; };

    void normalize();



};


