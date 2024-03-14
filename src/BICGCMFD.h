#pragma once
#include "BICGSolver.h"
#include "CMFD.h"
#include "JacobiBicgSolver.h"
#include "LineJacobiBicgSolver.h"
#include "NodalCPU.h"
#include "SSORSolver.h"
class BICGCMFD : public CMFD {
protected:
    // BICGSolver* _ls;
    // JacobiBicgSolver* _ls;
    // LineJacobiBicgSolver* _ls;
    SSORSolver* _ls;

    NodalCPU* _nodal;

    int   _nmaxbicg;
    float _epsbicg;

    double* _unshifted_diag;
    float   _eshift;

    int iter;

public:
    BICGCMFD(Geometry& g, CrossSection& x);
    virtual ~BICGCMFD();

    __host__ void init() override;
    __host__ void upddtil() override;
    __host__ void upddhat(double* flux, double* jnet) override;
    __host__ void setls(const double& eigv) override;
    __host__ void updjnet(double* flux, double* jnet) override;
    __host__ void updpsi(const double* flux) override;
    __host__ void drive(double& eigv, double* flux, double& errl2) override;

    __host__ void updnodal(double& _reigv, double* flux, double* jnet, double* phis);

    __host__ void            resetIteration();
    __host__ __device__ void setEshift(float eshift0);
    __host__ void            updls(const double& reigvs);

    __host__ __device__ void setls(const int& l);
    __host__ __device__ void updls(const int& l, const double& reigvs);

    __host__ void               axb(double* flux, double* aflux);
    __host__ double             residual(const double& reigv, const double& reigvs, const double* flux);
    __host__ void               wiel(const int& icy, const double* flux, double& reigvs, double& eigv, double& reigv, double& errl2);
    __host__ __device__ double& unshifted_diag(const int& igs, const int& ige, const int& l) {
        return _unshifted_diag[l * _g.ng2() + ige * _g.ng() + igs];
    };

    __host__ __device__ float& eshift() {
        return _eshift;
    };
};
