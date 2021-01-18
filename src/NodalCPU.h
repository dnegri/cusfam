#pragma once
#include "Nodal.h"
#include "CrossSection.h"

class NodalCPU : public Nodal {
private:
    CrossSection& xs;

public:
    NodalCPU(Geometry& g, CrossSection& xs);

    virtual ~NodalCPU();

    inline float& jnet(const int& ig, const int& lks) { return _jnet[lks * ng() + ig]; };
    inline double& flux(const int& ig, const int& lk) { return _flux[lk * ng() + ig]; };

    void init();
    void reset(CrossSection& xs, double* reigv, double* jnet, double* phif);
    void drive();

};


