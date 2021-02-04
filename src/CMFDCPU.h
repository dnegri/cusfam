#pragma once
#include "pch.h"
#include "CMFD.h"

class CMFDCPU : public CMFD {

private:
    CSRSolver* _ls;
public:
    CMFDCPU(Geometry &g, CrossSection &x);

    virtual ~CMFDCPU();

    __host__ __device__ void upddtil() override;
    __host__ __device__ void upddhat(double* flux, double* jnet) override;
    __host__ __device__ void setls() override;
    __host__ __device__ void updls(const double& reigvs);
    __host__ __device__ void updjnet(double* flux, double* jnet) override;
    __host__ __device__ void setls(const int &l);

    __host__ __device__ void updls(const int& l, const double& reigvs);

    __host__ __device__ void drive(double& eigv, double* flux, double* psi, float& errl2);
    __host__ __device__ double residual(const double& reigv, const double& reigvs, double* flux, double* psi);
    __host__ __device__ void axb(double* flux, double* aflux);
    __host__ __device__ double wiel(const int& icy, double* flux, double* psi, double& eigv, double& reigv, double& reigvs);
};


