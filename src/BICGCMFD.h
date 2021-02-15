#pragma once
#include "CMFD.h"
#include "BICGSolver.h"
#include "JacobiBicgSolver.h"
#include "NodalCPU.h"

class BICGCMFD : public CMFD {
protected:
//    BICGSolver* _ls;
    JacobiBicgSolver* _ls;
    NodalCPU* _nodal;

    int _nmaxbicg;
    float _epsbicg;

    CMFD_VAR* _unshifted_diag;
    float _eshift;

    int iter;


public:
    BICGCMFD(Geometry &g, CrossSection &x);
    virtual ~BICGCMFD();

    __host__ void init() override;
    __host__ void upddtil() override;
    __host__ void upddhat(SOL_VAR* flux, SOL_VAR* jnet) override;
    __host__ void setls(const double& eigv) override;
    __host__ void updjnet(SOL_VAR* flux, SOL_VAR* jnet) override;
    __host__ void updpsi(const SOL_VAR* flux) override;
    __host__ void drive(double& eigv, SOL_VAR* flux, float& errl2) override;

    __host__ void updnodal(double& _reigv, SOL_VAR* flux, SOL_VAR* jnet);


    __host__ void resetIteration();
    __host__ void setEshift(float eshift0);
    __host__ void updls(const double& reigvs);

    __host__ __device__ void setls(const int& l);
    __host__ __device__ void updls(const int& l, const double& reigvs);

    __host__ void axb(SOL_VAR* flux, SOL_VAR* aflux);
    __host__ double residual(const double& reigv, const double& reigvs,const SOL_VAR* flux);
    __host__ void wiel(const int& icy, const SOL_VAR* flux, double& reigvs, double& eigv, double& reigv, float& errl2);
    __host__ __device__ CMFD_VAR& unshifted_diag(const int& igs, const int& ige, const int& l) { return _unshifted_diag[l * _g.ng2() + ige * _g.ng() + igs]; };

    __host__ __device__ float& eshift() { return _eshift; };

}; 


