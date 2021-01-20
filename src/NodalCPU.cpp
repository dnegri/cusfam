#include "NodalCPU.h"

NodalCPU::~NodalCPU() {

}

NodalCPU::NodalCPU(Geometry &g, CrossSection& xs) : Nodal(g), xs(xs) {

	
	_ng = new int();
    _ng2 = new int();
    _nxyz = new int();
    _nsurf = new int();
	*_ng = _g.ng();
	*_ng2 = *_ng * *_ng;
	*_nxyz = _g.nxyz();
	*_nsurf = _g.nsurf();

	_trlcff0 = new float[nxyz() * NDIRMAX * ng()];
	_trlcff1 = new float[nxyz() * NDIRMAX * ng()];
	_trlcff2 = new float[nxyz() * NDIRMAX * ng()];
	_eta1 = new float[nxyz() * NDIRMAX * ng()];
	_eta2 = new float[nxyz() * NDIRMAX * ng()];
	_m260 = new float[nxyz() * NDIRMAX * ng()];
	_m251 = new float[nxyz() * NDIRMAX * ng()];
	_m253 = new float[nxyz() * NDIRMAX * ng()];
	_m262 = new float[nxyz() * NDIRMAX * ng()];
	_m264 = new float[nxyz() * NDIRMAX * ng()];
	_diagDI = new float[nxyz() * NDIRMAX * ng()];
	_diagD = new float[nxyz() * NDIRMAX * ng()];
	_dsncff2 = new float[nxyz() * NDIRMAX * ng()];
	_dsncff4 = new float[nxyz() * NDIRMAX * ng()];
	_dsncff6 = new float[nxyz() * NDIRMAX * ng()];
	_mu = new float[nxyz() * NDIRMAX * ng2()];
	_tau = new float[nxyz() * NDIRMAX * ng2()];
	_matM = new float[nxyz() * ng2()];
	_matMI = new float[nxyz() * ng2()];
	_matMs = new float[nxyz() * ng2()];
	_matMf = new float[nxyz() * ng2()];

	_xsnf = &xs.xsnf(0, 0);
	_xsdf = &xs.xsdf(0, 0);
	_xstf = &xs.xstf(0, 0);
	_chif = &xs.chif(0, 0);
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

void NodalCPU::reset(CrossSection& xs, double& reigv, float* jnet, double* phif) {
	_flux = phif;
	_jnet = jnet;
	_reigv = reigv;
}

void NodalCPU::drive(float* jnet) {

	for (int lk = 0; lk < nxyz(); ++lk) {
		updateConstant(lk);
		caltrlcff0(lk);
	}

	for (int lk = 0; lk < nxyz(); ++lk) {
		caltrlcff12(lk);
		updateMatrix(lk);
		calculateEven(lk);
	}

	for (int ls = 0; ls < nsurf(); ++ls) {
		calculateJnet(ls);
	}

	fflush(stdout);
}
