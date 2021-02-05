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

    __host__ __device__ void updpsi(const double* flux) override;

    __host__ __device__ void upddtil() override;
    __host__ __device__ void upddhat(double* flux, double* jnet) override;
    __host__ __device__ void setls(const double& eigv) override;
    __host__ __device__ void updjnet(double* flux, double* jnet) override;
    __host__ __device__ void setls(const int &l);

    __host__ __device__ void updls(const double& reigvs);
    __host__ __device__ void updls(const int& l, const double& reigvs);

    __host__ __device__ void drive(double& eigv, double* flux, float& errl2) override;
    __host__ __device__ double residual(const double& reigv, const double& reigvs, double* flux);
    __host__ __device__ void axb(double* flux, double* aflux);
    __host__ __device__ double wiel(const int& icy, double* flux, double& eigv, double& reigv, double& reigvs);
    __host__ __device__ void setEshift(float eshift0);
    __host__ __device__ double& eshift_diag(const int& igs, const int& ige, const int& l) { return _eshift_diag[l * _g.ng2() + ige * _g.ng() + igs]; };
};


