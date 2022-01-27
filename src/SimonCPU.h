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
    float _crit_nodal = 1.E-1;
    float _crit_flux = 1.E-5;
	float _crit_eigv = 1.E-6;
	float _crit_tm = 1.E-3;
	float _crit_tf = 1.E-3;
	float _crit_ppm = 1.E-1;

	bool _iter_new = true;

public:
    SimonCPU();
    virtual ~SimonCPU();

    inline BICGCMFD& cmfd() { return *_cmfd; }
    
	void setBurnup(const char* dir_burn, const float& burnup) override;    
	void initialize(const char* dbfile) override;
    void runKeff(const int& nmaxout) override;
    void runECP(const int& nmaxout, const double& eigvt) override;
    void runDepletion(const DepletionOption& option) override;
    void runXenonTransient(const DepletionOption& option) override;
    void normalize() override;

    void runSteady(const SteadyOption& condition) override;
	void runSteadySfam(const SteadyOption& condition) ;

    float updatePPM(const bool& first, const double& eigvt, const float& ppm, const float& ppmd, const double& eigv, const double& eigvd);

    void updateCriteria(const float& crit_flux);
};


