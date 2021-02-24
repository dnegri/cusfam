#pragma once
#include "Nodal.h"
#include "CrossSection.h"

class NodalCPU : public Nodal {
private:
    CrossSection& xs;

public:
    NodalCPU(Geometry& g, CrossSection& xs);

    virtual ~NodalCPU();

    inline SOL_VAR& jnet(const int& ig, const int& lks) { return _jnet[lks * ng() + ig]; };
    inline SOL_VAR& flux(const int& ig, const int& lk) { return _flux[lk * ng() + ig]; };

    void init();
    void reset(CrossSection& xs, const double& reigv, SOL_VAR* jnet, SOL_VAR* phif);
    void drive(SOL_VAR* jnet);

};


