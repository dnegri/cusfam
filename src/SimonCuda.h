#pragma once

#include "pch.h"
#include "CrossSection.h"
#include "Geometry.h"
#include "SteamTable.h"
#include "NodalCPU.h"
#include "CMFDCPU.h"
#include "BICGCMFD.h"
#include "Geometry.h"
#include "Depletion.h"
#include "CrossSection.h"
#include "Feedback.h"
#include "Simon.h"


class SimonCuda : public Simon {
private:
    BICGCMFD* _cmfd;
public:
    SimonCuda();
    virtual ~SimonCuda();

    inline BICGCMFD& cmfd() { return *_cmfd; }
    
    void initialize(const char* dbfile);
    void runKeff(const int& nmaxout);
    void runECP(const int& nmaxout, const double& eigvt);
    void runDepletion(const float& dburn);
    void runXenonTransient();
    void normalize();

};


