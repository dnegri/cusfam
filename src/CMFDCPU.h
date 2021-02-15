#pragma once
#include "pch.h"
#include "CMFD.h"

class CMFDCPU : public CMFD {

private:
    CSRSolver* _ls;

    double* _eshift_diag;
    float _eshift;

public:
    CMFDCPU(Geometry &g, CrossSection &x);

    virtual ~CMFDCPU();

    __host__ void updpsi(const SOL_VAR* flux) override;

    __host__ void upddtil() override;
    __host__ void upddhat(SOL_VAR* flux, SOL_VAR* jnet) override;
    __host__ void setls(const double& eigv) override;
    __host__ void updjnet(SOL_VAR* flux, SOL_VAR* jnet) override;
    __host__ __device__ void setls(const int &l);

    __host__ void updls(const double& reigvs);
    __host__ __device__ void updls(const int& l, const double& reigvs);

    __host__ void drive(double& eigv, SOL_VAR* flux, float& errl2) override;
    __host__ double residual(const double& reigv, const double& reigvs, SOL_VAR* flux);
    __host__ void axb(SOL_VAR* flux, SOL_VAR* aflux);
    __host__ double wiel(const int& icy, SOL_VAR* flux, double& eigv, double& reigv, double& reigvs);
    __host__ __device__ void setEshift(float eshift0);
    __host__ __device__ double& eshift_diag(const int& igs, const int& ige, const int& l) { return _eshift_diag[l * _g.ng2() + ige * _g.ng() + igs]; };
};


