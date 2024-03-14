#pragma once

#include "Geometry.h"
#include "pch.h"

class LineJacobiBicgSolver : public Managed {
protected:
    Geometry* _g;

    double _calpha, _cbeta, _crho, _comega;

    double *_vy, *_vz;

    double *_vr, *_vr0, *_vp, *_vv, *_vs, *_vt;

    double* _delinv;
    double* _delcc;

public:
    LineJacobiBicgSolver(){};

    LineJacobiBicgSolver(Geometry& g);

    virtual ~LineJacobiBicgSolver();

    virtual void reset(double* diag, double* cc, double* flux, double* src, double& r20);

    virtual void minv(double* cc, double* b, double* x);

    virtual void facilu(double* diag, double* cc);

    virtual void axb(double* diag, double* cc, double* flux, double* aflux);

    virtual void solve(double* diag, double* cc, double& r20, double* flux, double& r2);

    __host__ __device__ double reset(const int& l, double* diag, double* cc, double* flux, double* src);

    __host__ __device__ void minv(const int& j, const int& k, double* cc, double* b, double* x);

    __host__ __device__ void facilu(const int& j, const int& k, double* diag, double* cc);

    __host__ __device__ double axb(const int& ig, const int& l, double* diag, double* cc, double* flux);

    __host__ __device__ double& alpha() {
        return _calpha;
    }

    __host__ __device__ double& beta() {
        return _cbeta;
    }

    __host__ __device__ double& rho() {
        return _crho;
    }

    __host__ __device__ double& omega() {
        return _comega;
    }

    __host__ __device__ Geometry& g() {
        return *_g;
    }

    __host__ __device__ double& delinv(const int& igs, const int& ige, const int& l) {
        return _delinv[(l * _g->ng2()) + (ige)*_g->ng() + (igs)];
    };

    __host__ __device__ double& delcc(const int& igs, const int& ige, const int& l) {
        return _delcc[(l * _g->ng2()) + (ige)*_g->ng() + (igs)];
    };

    double& vr(int ig, int l) {
        return _vr[(l * _g->ng()) + ig];
    };

    double& vr0(int ig, int l) {
        return _vr0[(l * _g->ng()) + ig];
    };

    double& vp(int ig, int l) {
        return _vp[(l * _g->ng()) + ig];
    };

    double& vv(int ig, int l) {
        return _vv[(l * _g->ng()) + ig];
    };
};
