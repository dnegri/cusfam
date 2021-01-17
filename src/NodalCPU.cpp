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
    _mu = new float[ _nxyz * NDIRMAX * _ng2];
    _tau = new float[ _nxyz * NDIRMAX * _ng2];
    _matM = new float[ _nxyz * _ng2];
    _matMI = new float[ _nxyz * _ng2];
    _matMs = new float[ _nxyz * _ng2];
    _matMf = new float[ _nxyz * _ng2];

	_xsnff = &xs.xsnf(0, 0);
	_xsdf = &xs.xsdf(0, 0);
	_xstf = &xs.xstf(0, 0);
	_xschif = &xs.chif(0, 0);
	_xsadf = &xs.xsadf(0, 0);
	_xssf = &xs.xssf(0, 0, 0);
	_neib = &g.neib(0, 0);
	_lktosfc = &g.lktosfc(0,0, 0);
	_hmesh = &g.hmesh(0, 0);

	_lklr = &g.lklr(0, 0);
	_sgnlr = &g.sgnlr(0, 0);
	_idirlr = &g.idirlr(0, 0);
	_albedo = &g.albedo(0, 0);

}

void NodalCPU::init() {

}

void NodalCPU::reset(CrossSection &xs, double *reigv, double *jnet, double *phif) {
	for (size_t ls = 0; ls < _nsurf; ls++)
	{
		int idirl = _g.idirlr(LEFT, ls);
		int idirr = _g.idirlr(RIGHT, ls);
		int lkl = _g.lklr(LEFT, ls);
		int lkr = _g.lklr(RIGHT, ls);
		int kl = lkl / _g.nxy();
		int ll = lkl % _g.nxy();
		int kr = lkr / _g.nxy();
		int lr = lkr % _g.nxy();


		for (size_t ig = 0; ig < _ng; ig++)
		{
			if (lkr < 0) {
				int idx =
					idirl * (_g.nz() * _g.nxy() * _g.ng() * LR)
					+ kl * (_g.nxy() * _g.ng() * LR)
					+ ll * (_g.ng() * LR)
					+ ig * LR
					+ RIGHT;
				this->jnet(ig, ls) = jnet[idx];
			}
			else {
				int idx =
					idirr * (_g.nz() * _g.nxy() * _g.ng() * LR)
					+ kr * (_g.nxy() * _g.ng() * LR)
					+ lr * (_g.ng() * LR)
					+ ig * LR
					+ LEFT;
				this->jnet(ig, ls) = jnet[idx];
			}
		}
	}

	int lk = -1;
	for (size_t k = 0; k < _g.nz(); k++)
	{
		for (size_t l = 0; l < _g.nxy(); l++)
		{
			lk++;
			for (size_t ig = 0; ig < _g.ng(); ig++)
			{
				int idx = (k + 1) * (_g.nxy() + 1) * _g.ng() + (l + 1) * _g.ng() + ig;
				this->flux(ig, lk) = phif[idx];
			}
		}
	}

	_reigv = *reigv;
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
        ::sanm2n_calculateJnet (ls, _ng, _ng2, _nsurf, _lklr, _idirlr, _sgnlr, _albedo, _hmesh, _xsadf, _m251,_m253, _m260, _m262, _m264,_diagD, _diagDI, _matM, _matMI, _flux, _trlcff0, _trlcff1,_trlcff2, _mu, _tau, _eta1, _eta2, _dsncff2, _dsncff4, _dsncff6, _jnet);
    }    
}
