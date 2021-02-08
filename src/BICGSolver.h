#pragma once

#include "Geometry.h"

class BICGSolver {
private:
    Geometry &_g;

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
    , *_y1d
    , *_b1i
    , *_b01d
    , *_s1dl
    , *_b03d
    , *_s3d
    , *_s3dd
    ;

    CMFD_VAR *_del
    , *_ainvd
    , *_ainvl
    , *_ainvu
    , *_au
    , *_delinv
    , *_al
    , *_deliau
    ;
public:
    BICGSolver(Geometry &g);

    virtual ~BICGSolver();

    void reset(CMFD_VAR *diag, CMFD_VAR *cc, double *phi, CMFD_VAR *src, double &r20);
    double reset(const int& ig, const int& l, CMFD_VAR *diag, CMFD_VAR *cc, double *phi, CMFD_VAR *src);

    void sol1d(const int &j, const int &k, CMFD_VAR *b, CMFD_VAR *x);

    void sol2d(CMFD_VAR *cc, const int &k, CMFD_VAR *b, CMFD_VAR *x);

    void minv(CMFD_VAR *cc, CMFD_VAR *b, double *x);

    void abi1d(const int &j, const int &k);

    void facilu1d(const int &j, const int &k);

    void facilu(CMFD_VAR *diag, CMFD_VAR *cc);

    void solve(CMFD_VAR *diag, CMFD_VAR *cc, double &r20, double *phi, double &r2);

    void axb(CMFD_VAR *diag, CMFD_VAR *cc, double *phi, CMFD_VAR *aphi);
    double axb(const int& ig, const int& l, CMFD_VAR *diag, CMFD_VAR *cc, double *phi);

};


