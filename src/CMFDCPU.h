#pragma once
#include "pch.h"
#include "CMFD.h"

class CMFDCPU : public CMFD {

private:

public:
    CMFDCPU(Geometry &g, CrossSection &x);

    virtual ~CMFDCPU();

    void upddtil() override;
    void upddhat(double* flux, float* jnet) override;
    void setls() override;
    void updls(const double& reigvs);

    void setls(const int &l);
    void updls(const int& l, const double& reigvs);

    void drive(double& eigv, double* flux, double* psi, float& errl2);
    double residual(const double& reigv, const double& reigvs, double* flux, double* psi);
    void axb(double* flux, double* aflux);
    double wiel(const int& icy, double* flux, double* psi, double& eigv, double& reigv, double& reigvs);
};


