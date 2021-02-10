#pragma once
#include "pch.h"
#include "Geometry.h"

class JacobiBicgSolver : public Managed {
protected:
    Geometry *_g;

    CMFD_VAR _calpha
    , _cbeta
    , _crho
    , _comega;

    double *_vy
    , *_vz;


    CMFD_VAR *_vr
    , *_vr0
    , *_vp
    , *_vv
    , *_vs
    , *_vt
    ;

    CMFD_VAR *_delinv;
public:
    JacobiBicgSolver() {};
    JacobiBicgSolver(Geometry &g);
    virtual ~JacobiBicgSolver();

    void reset(CMFD_VAR *diag, CMFD_VAR *cc, double *phi, CMFD_VAR *src, CMFD_VAR& r20);
    void minv(CMFD_VAR* cc, CMFD_VAR* b, double* x);
    void facilu(CMFD_VAR* diag, CMFD_VAR* cc);
    void axb(CMFD_VAR* diag, CMFD_VAR* cc, double* phi, CMFD_VAR* aphi);

    __host__ __device__ float reset(const int& l, CMFD_VAR* diag, CMFD_VAR* cc, double* phi, CMFD_VAR* src);
    __host__ __device__ void minv(const int& l, CMFD_VAR* cc, CMFD_VAR* b, double* x);
    __host__ __device__ void facilu(const int& l, CMFD_VAR* diag, CMFD_VAR* cc);
    __host__ __device__ CMFD_VAR axb(const int& ig, const int& l, CMFD_VAR* diag, CMFD_VAR* cc, double* phi);

    void solve(CMFD_VAR *diag, CMFD_VAR *cc, CMFD_VAR& r20, double *phi, double &r2);

    
    __host__ __device__ CMFD_VAR& alpha() { return _calpha; }
    __host__ __device__ CMFD_VAR& beta() { return _cbeta; }
    __host__ __device__ CMFD_VAR& rho() { return _crho; }
    __host__ __device__ CMFD_VAR& omega() { return _comega; }
    __host__ __device__ Geometry& g() { return *_g; }

    __host__ __device__ CMFD_VAR& delinv(const int& igs, const int& ige, const int& l) {
        return _delinv[(l * _g->ng2()) + (ige)*_g->ng() + (igs)];
    };

};


