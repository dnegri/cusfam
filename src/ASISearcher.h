#pragma once

#include "Geometry.h"
#include "CrossSection.h"
#include "JIArray.h"
#include "Simon.h"

class ASISearcher {
private:

    Geometry& g;

    double2 delmac;

    int ndiv;
    double1 xediv, xen1d, xeratio;
    float1 p1dp, p1d;
    double1 xeac, xebc, coeff;
    double2 fintg;

    double xeavg, xemid, xe23a, xetmb, delxe;

    double xextrp, exthght;

    double ndavg, ndcnt, ndcnt2, ndtmb, nddel;
    double asi0, asip;

public:
    ASISearcher(Geometry &g_);
    virtual ~ASISearcher();
    
    void reset(Depletion& d);
    void search(const double &epsflx1, const int &nouter, Simon &simon, const SteadyOption &option, const double &targetASI, const double &epsASI);
    void divideXenon(Depletion& d);
    void generateXenonShape2(double1 &xef);
    void changeXenonDensity(const double1 &xef, Depletion& d, CrossSection& x);
    void calculateIntegralForXeDiv();
    void getIntegral(const double& x, double1& fintg1);
    void invmatxvec1(double2 &mat, double1 &vec, double1 &sol, const int &n);
};
