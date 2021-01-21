#pragma once
#include "pch.h"
#include "Geometry.h"
#include "CrossSection.h"

class BICG : public Managed {
private:
    Geometry _g;
    CrossSection _x;

    double _calpha;
    double _cbeta;
    double _crho;
    double _comega;
    //(_4,nx)
    float * _del, _au, _ainvl, _ainvu, _ainvd;

    //(_4,_nxy,nz)
    float * _delinv, _al, deliau;

    //(_ng,_nxy,nz)
    float * _vr,_vr0,_vp,_vv,_vs,_vt,_vy,_vz;
public:
    BICG(const Geometry &g, const CrossSection &x);
    virtual ~BICG();

    void facilu2g();
    void facilu1d2g(const int& irow,const int& k);
    void abi1d2g(const int& irow,const int& k);
    void initbicg2g(float* phi, float* rhs, float* r20);
    void solbicg2g(float& r20, float& r2,float* phi);
    void minv2g(float* b, float* x);
    void sol1d2g(const int& irow,const int& k,float* b,float* x);
    void sol2d2g(const int& k,float* b,float* x);
    void solbicg2g(float& r20,float& r2)
};


