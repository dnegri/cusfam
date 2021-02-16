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


