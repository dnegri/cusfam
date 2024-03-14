#pragma once

#include "JIArray.h"
#include "CrossSection.h"
#include "NodalCPU.h"
#include "BICGCMFD.h"

using namespace dnegri::jiarray;

class ShapeMatch {
    int maxiter;
    double1 pow1d;
    double1 vol1d;

    double2 dxsa;
    double2 xmacp;
    double2 lkg1d;
    double2 xstf1d0;
    double2 diffa;
    double2 phi1d;
    double2 jnet1d;
    double2 phis1d;

    double2 crntol;
    double2 crntor;
    double2 crntil;
    double2 crntir;

    double2 c1gu;
    double2 c2gu;
    double2 c3gu;
    double2 c4gu;
    double2 c5gu;
    double2 c6gu;
    double2 a1gu;
    double2 a2gu;
    double2 a3gu;
    double2 a4gu;


    CrossSection& xs;
    Geometry& g;
    Geometry g1d;
    CrossSection xs1d;

    BICGCMFD* cmfd1d;

private:
    void updateAxialData(const double& eigv, const double2& flux, const double2& jnet, const double1& powshp);
    void neminit();
    void moment();
    void nemcoef();
    double fluxcal(const double1& powshp);
    void thrmabs();
    void currento();
    double totfis();

public:
    ShapeMatch(Geometry& g, CrossSection& xs);

    virtual ~ShapeMatch();

    void solve(const double& eigv, const double2& flux, const double2& jnet, const double1& powshp);

    const double2& getShapeXS() {return dxsa;}


};
