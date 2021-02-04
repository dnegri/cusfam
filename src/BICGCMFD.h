#pragma once
#include "CMFD.h"
#include "BICGSolver.h"

class BICGCMFD : public CMFD {
private:
    int _nmaxbicg;
    double _epsbicg;
    double* _eshift_diag;
    BICGSolver* _ls;
private:
    double& eshift_diag(const int& igs, const int& ige, const int& l) {return _eshift_diag[l*_g.ng2()+ige*_g.ng()+igs];};
public:
    BICGCMFD(Geometry &g, CrossSection &x);
    virtual ~BICGCMFD();

    __host__ __device__     void upddtil() override;
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


