#include "NodalCPU.h"
#include "sanm2n.h"

NodalCPU::~NodalCPU() {

}

NodalCPU::NodalCPU(Geometry &g, CrossSection& xs) : Nodal(g), xs(xs) {
    _ng = _g.ng();
    _ng2 = _ng * _ng;
    _nxyz = _g.nxyz();
    _nsurf = _g.nsurf();

    _jnet = new float[_nsurf * _ng];
    _flux = new double[_nxyz * _ng];

    _trlcff0 = new float[ _nxyz * NDIRMAX * _ng];
    _trlcff1 = new float[ _nxyz * NDIRMAX * _ng];
    _trlcff2 = new float[ _nxyz * NDIRMAX * _ng];
    _eta1 = new float[ _nxyz * NDIRMAX * _ng];
    _eta2 = new float[ _nxyz * NDIRMAX * _ng];
    _mu = new float[ _nxyz * NDIRMAX * _ng];
    _tau = new float[ _nxyz * NDIRMAX * _ng];
    _m260 = new float[ _nxyz * NDIRMAX * _ng];
    _m251 = new float[ _nxyz * NDIRMAX * _ng];
    _m253 = new float[ _nxyz * NDIRMAX * _ng];
    _m262 = new float[ _nxyz * NDIRMAX * _ng];
    _m264 = new float[ _nxyz * NDIRMAX * _ng];
    _diagDI = new float[ _nxyz * NDIRMAX * _ng];
    _diagD = new float[ _nxyz * NDIRMAX * _ng];
    _dsncff2 = new float[ _nxyz * NDIRMAX * _ng];
    _dsncff4 = new float[ _nxyz * NDIRMAX * _ng];
    _dsncff6 = new float[ _nxyz * NDIRMAX * _ng];
    _xstf = new XS_PRECISION[ _nxyz * _ng];
    _xsdf = new XS_PRECISION[ _nxyz * _ng];
    _xsnff = new XS_PRECISION[ _nxyz * _ng];
    _xschif = new XS_PRECISION[ _nxyz * _ng];
    _xsadf = new XS_PRECISION[ _nxyz * _ng];
    _xssf = new XS_PRECISION[ _nxyz * _ng2];
    _matM = new float[ _nxyz * _ng2];
    _matMI = new float[ _nxyz * _ng2];
    _matMs = new float[ _nxyz * _ng2];
    _matMf = new float[ _nxyz * _ng2];
}

void NodalCPU::init() {

}

void NodalCPU::reset(CrossSection &xs, double *reigv, double *jnet, double *phif) {

}

void NodalCPU::drive() {

    for (int lk = 0; lk < _nxyz; ++lk) {
        ::sanm2n_reset (lk, _ng, _ng2, _nxyz, _hmesh, _xstf, _xsdf, _eta1, _eta2, _m260, _m251, _m253, _m262, _m264, _diagD, _diagDI);
        ::sanm2n_calculateTransverseLeakage(lk, _ng, _ng2, _nxyz, _lktosfc, _idirlr, _neib, _hmesh, _albedo, _jnet, _trlcff0, _trlcff1, _trlcff2);
        ::sanm2n_resetMatrix (lk, _ng, _ng2, _nxyz, _reigv, _xstf, _xsnff, _xschif, _xssf, _matMs, _matMf, _matM);
        ::sanm2n_prepareMatrix (lk, _ng, _ng2, _nxyz, _m251, _m253, _diagD, _diagDI, _matM, _matMI, _tau, _mu);
        ::sanm2n_calculateEven (lk, _ng, _ng2, _nxyz, _m260, _m262, _m264, _diagD, _diagDI, _matM, _flux, _trlcff0, _trlcff2, _dsncff2, _dsncff4, _dsncff6);
    }

    for (int ls = 0; ls < _nsurf; ++ls) {
        ::sanm2n_calculateJnet (ls, _ng, _ng2, _nsurf, _lklr, _idirlr, _sgnlr, _hmesh, _xsadf, _m260, _m262, _m264,_diagD, _diagDI, _matM, _matMI, _flux, _trlcff0, _trlcff1,_trlcff2, _mu, _tau, _eta1, _eta2, _dsncff2, _dsncff4, _dsncff6, _jnet);
    }    
}
