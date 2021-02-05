#pragma once
#include "CMFD.h"
#include "BICGSolver.h"

class BICGCMFD : public CMFD {
private:
    BICGSolver* _ls;

    int _nmaxbicg;
    double _epsbicg;

    double* _eshift_diag;
    float _eshift;


public:
    BICGCMFD(Geometry &g, CrossSection &x);
    virtual ~BICGCMFD();

    __host__ __device__ void upddtil() override;
    __host__ __device__ void upddhat(double* flux, double* jnet) override;
    __host__ __device__ void setls(const double& eigv) override;
    __host__ __device__ void updjnet(double* flux, double* jnet) override;
    __host__ __device__ void updpsi(const double* flux) override;
    __host__ __device__ void drive(double& eigv, double* flux, float& errl2) override;


    __host__ __device__ void setEshift(float eshift0);
    __host__ __device__ void setls(const int &l);
    __host__ __device__ void updls(const double& reigvs);
    __host__ __device__ void updls(const int& l, const double& reigvs);

    __host__ __device__ void axb(double* flux, double* aflux);
    __host__ __device__ double residual(const double& reigv, const double& reigvs,const double* flux);
    __host__ __device__ void wiel(const int& icy, const double* flux, double& reigvs, double& eigv, double& reigv, float& errl2);
    __host__ __device__ double& eshift_diag(const int& igs, const int& ige, const int& l) { return _eshift_diag[l * _g.ng2() + ige * _g.ng() + igs]; };

};


