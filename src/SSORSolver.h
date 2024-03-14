#pragma once

#include "Geometry.h"
#include "pch.h"

class SSORSolver : public Managed {
protected:
    Geometry* _g;

    double _rhoJacobi, _rhoGS, _omega;
    double* _src;

    double* _delinv;
public:
    SSORSolver(){};

    SSORSolver(Geometry& g);

    virtual ~SSORSolver();

    virtual void reset(double* diag, double* cc, double* flux, double* src, double& r20);

    virtual void minv(double* cc, double* b, double* x, double& errl2);

    virtual void facilu(double* diag, double* cc);

    virtual void axb(double* diag, double* cc, double* flux, double* aflux);

    virtual void solve(double* diag, double* cc, double& r20, double* flux, double& r2);

    __host__ __device__ double reset(const int& l, double* diag, double* cc, double* flux, double* src);

    __host__ __device__ void minv(const int& l, double* cc, double* b, double* x, double& errl2);

    __host__ __device__ void facilu(const int& l, double* diag, double* cc);

    __host__ __device__ double axb(const int& ig, const int& l, double* diag, double* cc, double* flux);


    __host__ __device__ Geometry& g() {
        return *_g;
    }

    __host__ __device__ double& delinv(const int& igs, const int& ige, const int& l) {
        return _delinv[(l * _g->ng2()) + (ige)*_g->ng() + (igs)];
    };
};
