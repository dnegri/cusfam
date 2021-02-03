#pragma once

#include "Geometry.h"

class BICGSolver {
private:
    Geometry &_g;

    double _calpha
    , _cbeta
    , _crho
    , _comega;

    double *_vr
    , *_vr0
    , *_vp
    , *_vv
    , *_vs
    , *_vt
    , *_vy
    , *_vz
    , *_y1d
    , *_b1i
    , *_b01d
    , *_s1dl
    , *_b03d
    , *_s3d
    ;

    double *_del
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

    void reset(double *diag, double *cc, double *phi, double *src, double &r20);
    double reset(const int& ig, const int& l, double *diag, double *cc, double *phi, double *src);

    void sol1d(const int &j, const int &k, double *b, double *x);

    void sol2d(double *cc, const int &k, double *b, double *x);

    void minv(double *cc, double *b, double *x);

    void abi1d(const int &j, const int &k);

    void facilu1d(const int &j, const int &k);

    void facilu(double *diag, double *cc);

    void solve(double *diag, double *cc, double &r20, double *phi, double &r2);

    void axb(double *diag, double *cc, double *phi, double *aphi);
    double axb(const int& ig, const int& l, double *diag, double *cc, double *phi);

};


