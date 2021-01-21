#pragma once
#include "pch.h"
#include "CMFD.h"

class CMFDCPU : public CMFD {

public:
    CMFDCPU(Geometry &g, CrossSection &x);

    virtual ~CMFDCPU();

    void upddtil() override;
    void upddhat() override;
    void setls() override;

    void drive(double& eigv, float* flux, float& errl2);
    double residual(const double& reigv, const double& reigvs, float* flux);
    void axb(float* flux, double* aflux);
};


