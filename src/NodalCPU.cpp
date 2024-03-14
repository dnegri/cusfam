#include "NodalCPU.h"

NodalCPU::~NodalCPU() {
}

NodalCPU::NodalCPU(Geometry& g, CrossSection& xs)
    : Nodal(g), xs(xs) {

    _ng     = new int();
    _ng2    = new int();
    _nxyz   = new int();
    _nsurf  = new int();
    *_ng    = _g.ng();
    *_ng2   = *_ng * *_ng;
    *_nxyz  = _g.nxyz();
    *_nsurf = _g.nsurf();

    _trlcff0 = new double[nxyz() * NDIRMAX * ng()];
    _trlcff1 = new double[nxyz() * NDIRMAX * ng()];
    _trlcff2 = new double[nxyz() * NDIRMAX * ng()];
    _eta1    = new double[nxyz() * NDIRMAX * ng()];
    _eta2    = new double[nxyz() * NDIRMAX * ng()];
    _m260    = new double[nxyz() * NDIRMAX * ng()];
    _m251    = new double[nxyz() * NDIRMAX * ng()];
    _m253    = new double[nxyz() * NDIRMAX * ng()];
    _m262    = new double[nxyz() * NDIRMAX * ng()];
    _m264    = new double[nxyz() * NDIRMAX * ng()];
    _diagDI  = new double[nxyz() * NDIRMAX * ng()];
    _diagD   = new double[nxyz() * NDIRMAX * ng()];
    _dsncff2 = new double[nxyz() * NDIRMAX * ng()];
    _dsncff4 = new double[nxyz() * NDIRMAX * ng()];
    _dsncff6 = new double[nxyz() * NDIRMAX * ng()];
    _mu      = new double[nxyz() * NDIRMAX * ng2()];
    _tau     = new double[nxyz() * NDIRMAX * ng2()];
    _matM    = new double[nxyz() * ng2()];
    _matMI   = new double[nxyz() * ng2()];
    _matMs   = new double[nxyz() * ng2()];
    _matMf   = new double[nxyz() * ng2()];

    _xsnf    = &xs.xsnf(0, 0);
    _xsdf    = &xs.xsdf(0, 0);
    _xstf    = &xs.xstf(0, 0);
    _chif    = &xs.chif(0, 0);
    _xsadf   = &xs.xsadf(0, 0);
    _xssf    = &xs.xssf(0, 0, 0);
    _neib    = &g.neib(0, 0);
    _lktosfc = &g.lktosfc(0, 0, 0);
    _hmesh   = &g.hmesh(0, 0);

    _lklr   = &g.lklr(0, 0);
    _sgnlr  = &g.sgnlr(0, 0);
    _idirlr = &g.idirlr(0, 0);
    _albedo = &g.albedo(0, 0);
}

void NodalCPU::init() {
}

void NodalCPU::reset(CrossSection& xs, const double& reigv, double* jnet, double* phif, double* phis) {
    _flux  = phif;
    _jnet  = jnet;
    _phis  = phis;
    _reigv = reigv;
}

void NodalCPU::drive(double* jnet) {

    // printf("updateConstant\n");
#pragma omp parallel for
    for (int lk = 0; lk < nxyz(); ++lk) {
        updateConstant(lk);
    }
    // printf("caltrlcff0\n");
#pragma omp parallel for
    for (int lk = 0; lk < nxyz(); ++lk) {
        caltrlcff0(lk);
    }

    // printf("caltrlcff12\n");
#pragma omp parallel for
    for (int lk = 0; lk < nxyz(); ++lk) {
        caltrlcff12(lk);
    }
    // printf("updateMatrix\n");
#pragma omp parallel for
    for (int lk = 0; lk < nxyz(); ++lk) {
        updateMatrix(lk);
    }
    // printf("calculateEven\n");
#pragma omp parallel for
    for (int lk = 0; lk < nxyz(); ++lk) {

        calculateEven(lk);
    }

    // printf("calculateJnet\n");
#pragma omp parallel for
    for (int ls = 0; ls < nsurf(); ++ls) {
        calculateJnet(ls);
    }
}
