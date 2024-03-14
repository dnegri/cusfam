#pragma once
#include "pch.h"
#include "BICGCMFD.h"

class BICGCMFDCuda : public BICGCMFD {
private:
    double* _gammad_dev;
    double* _gamman_dev;
    float* _errl2_dev;
    double* _psid;
public:
    BICGCMFDCuda(Geometry& g, CrossSection& x);
    virtual ~BICGCMFDCuda();

    __host__ void init() override;

    __host__ void upddtil() override;
    __host__ void upddhat(double* flux, double* jnet) override;
    __host__ void setls(const double& eigv) override;
    __host__ void updjnet(double* flux, double* jnet) override;
    __host__ void updpsi(const double* flux) override;
    __host__ void updsrc(const double& reigvdel);
    __host__ void drive(double& eigv, double* flux, float& errl2) override;

    __host__ void updls(const double& reigvs);

    __host__ void axb(double* flux, double* aflux);
    __host__ void axb1(double* flux, double* aflux);
    __host__ void wiel(const int& icy, const double* flux, double& reigvs, double& eigv, double& reigv, float& errl2);

    __host__ __device__ double& psid(const int& l) { return _psid[l]; };
};

