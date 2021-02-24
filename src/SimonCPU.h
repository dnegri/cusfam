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


class SimonCPU : public Simon {
private:
    BICGCMFD* _cmfd;
    float _crit_xenon = 1.E-4;
    float _crit_nodal = 1.E-4;
    float _crit_flux = 1.E-5;

public:
    SimonCPU();
    virtual ~SimonCPU();

    inline BICGCMFD& cmfd() { return *_cmfd; }
    
    void initialize(const char* dbfile) override;
    void runKeff(const int& nmaxout) override;
    void runECP(const int& nmaxout, const double& eigvt) override;
    void runDepletion(const float& tsec) override;
    void runXenonTransient() override;
    void normalize() override;

    void runSteady(const SteadyOption& condition) override;

    float updatePPM(const bool& first, const double& eigvt, const float& ppm, const float& ppmd, const double& eigv, const double& eigvd);

    void updateCriteria(const float& crit_flux);
};


