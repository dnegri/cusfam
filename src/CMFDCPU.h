#pragma once
#include "pch.h"
#include "CMFD.h"

class CMFDCPU : public CMFD {

private:

public:
    CMFDCPU(Geometry &g, CrossSection &x);

    virtual ~CMFDCPU();

    void upddtil() override;
    void upddhat() override;
    void setls() override;

    void drive(double& eigv, double* flux, float& errl2);
    double residual(const double& reigv, const double& reigvs, double* flux);
    void axb(double* flux, double* aflux);
    double wiel(const int& icy, double* flux, double* psi, double& eigv, double& reigv);
};


