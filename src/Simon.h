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

    CMFD* cmfd;

    int _nstep;
    float _epsbu;

    float* _bucyc;
    float* _power;
    double* _flux;
    double* _jnet;
    double reigv;
    double eigv;

public:

    Simon();

    virtual ~Simon();
    void initialize(const char* dbfile);
    void setBurnup(const float& burnup);
    void runStatic();
    void runDepletion(const float& dburn);
    void runXenonTransient();



};


