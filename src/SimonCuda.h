#pragma once

#include "pch.h"
#include "Geometry.h"
#include "GeometryCuda.h"
#include "CrossSection.h"
#include "CrossSectionCuda.h"
#include "SteamTableCuda.h"
#include "BICGCMFDCuda.h"
#include "DepletionCuda.h"
#include "FeedbackCuda.h"
#include "Simon.h"


class SimonCuda : public Simon {
private:
    GeometryCuda* _gcuda;
    CrossSectionCuda* _xcuda;
    FeedbackCuda* _fcuda;
    SteamTableCuda* _steamcuda;
    DepletionCuda* _dcuda;
    BICGCMFDCuda* _cmfdcuda;
    BICGCMFD* _cmfdcpu;

    SOL_VAR* _flux_cuda;
    SOL_VAR* _jnet_cuda;
    float* _power_cuda;
public:
    SimonCuda();
    virtual ~SimonCuda();

    inline BICGCMFD& cmfd() { return *_cmfdcuda; }
    
    void initialize(const char* dbfile);
    void setBurnup(const float& burnup);
    void runKeff(const int& nmaxout);
    void runECP(const int& nmaxout, const double& eigvt);
    void runDepletion(const float& dburn);
    void runXenonTransient();
    void normalize();

};


