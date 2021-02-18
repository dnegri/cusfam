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

    SOL_VAR *_vy
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

    virtual void reset(CMFD_VAR *diag, CMFD_VAR *cc, SOL_VAR* flux, CMFD_VAR *src, CMFD_VAR& r20);
    virtual void minv(CMFD_VAR* cc, CMFD_VAR* b, SOL_VAR* x);
    virtual void facilu(CMFD_VAR* diag, CMFD_VAR* cc);
    virtual void axb(CMFD_VAR* diag, CMFD_VAR* cc, SOL_VAR* flux, CMFD_VAR* aflux);
    virtual void solve(CMFD_VAR* diag, CMFD_VAR* cc, CMFD_VAR& r20, SOL_VAR* flux, CMFD_VAR& r2);

    __host__ __device__ float reset(const int& l, CMFD_VAR* diag, CMFD_VAR* cc, SOL_VAR* flux, CMFD_VAR* src);
    __host__ __device__ void minv(const int& l, CMFD_VAR* cc, CMFD_VAR* b, SOL_VAR* x);
    __host__ __device__ void facilu(const int& l, CMFD_VAR* diag, CMFD_VAR* cc);
    __host__ __device__ CMFD_VAR axb(const int& ig, const int& l, CMFD_VAR* diag, CMFD_VAR* cc, SOL_VAR* flux);

    __host__ __device__ CMFD_VAR& alpha() { return _calpha; }
    __host__ __device__ CMFD_VAR& beta() { return _cbeta; }
    __host__ __device__ CMFD_VAR& rho() { return _crho; }
    __host__ __device__ CMFD_VAR& omega() { return _comega; }
    __host__ __device__ Geometry& g() { return *_g; }

    __host__ __device__ CMFD_VAR& delinv(const int& igs, const int& ige, const int& l) {
        return _delinv[(l * _g->ng2()) + (ige)*_g->ng() + (igs)];
    };

};


