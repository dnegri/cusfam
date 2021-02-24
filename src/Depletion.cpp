#include "Depletion.h"
#include "myblas.h"

#define xsmica(ig, iiso, l) xsmica[(l) * _g.ng() * NISO + (iiso) * _g.ng() + (ig)]
#define xsmicf(ig, iiso, l) xsmicf[(l) * _g.ng() * NISO + (iiso) * _g.ng() + (ig)]
#define xsmic2n(ig, l) xsmic2n[(l) * _g.ng() + (ig)]
#define flux(ig, l) flux[(l) * _g.ng() + (ig)]
#define ati(iiso, l) ati[(l) * NISO + (iiso)]
#define atd(iiso, l) atd[(l) * NISO + (iiso)]
#define atavg(iiso, l) atavg[(l) * NISO + (iiso)]


Depletion::Depletion(Geometry& g) : _g(g) {

}

Depletion::~Depletion() {
	delete[] _nheavy;
	delete[] _hvyids;
	delete[] _reactype;
	delete[] _hvyupd;
	delete[] _dcy;
}

void Depletion::init()
{
	_nhvychn = 4;
	_nheavy = new int[4]{ 10, 7, 8, 2 };
	_ihvys = new int[4]{ 0, 10, 17, 25 };
	_nhvyids = 10 + 7 + 8 + 2;

	_hvyids = new int[_nhvyids] {
		U234, U235, U236, NP37, PU48, PU49, PU40, PU41, PU42, AM43,
			U238, NP39, PU49, PU40, PU41, PU42, AM43,
			U238, NP37, PU48, PU49, PU40, PU41, PU42, AM43,
			PU48, U234};

	_reactype = new int[_nhvyids] {
		R_CAP, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP,
			R_CAP, R_CAP, R_DEC, R_CAP, R_CAP, R_CAP, R_CAP,
			R_CAP, R_N2N, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP,
			R_CAP, R_DEC};

	_hvyupd = new int[_nhvyids] {
		UPDATE, UPDATE, UPDATE, UPDATE, UPDATE, LEAVE, LEAVE, LEAVE, LEAVE, LEAVE,
			UPDATE, UPDATE, UPDATE, UPDATE, UPDATE, UPDATE, UPDATE,
			HOLD, LEAVE, LEAVE, LEAVE, LEAVE, LEAVE, LEAVE, LEAVE,
			HOLD, LEAVE};

	_dcy = new float[NDEP] {};

	for (int idcy = 0; idcy < NDCY; ++idcy) {
		_dcy[ISODCY[idcy]] = DCY[idcy];
	}

	_cap = new float[NDEP * _g.nxyz()]{};
	_rem = new float[NDEP * _g.nxyz()]{};
	_fis = new float[NDEP * _g.nxyz()]{};

	_tn2n = new float[_g.nxyz()];

	_dnst = new float[NISO * _g.nxyz()]{};
	_dnst_new = new float[NISO * _g.nxyz()]{};
	_dnst_avg = new float[NISO * _g.nxyz()]{};
	_burn = new float[_g.nxyz()]{};
	_h2on = new float[_g.nxyz()]{};
	_buconf = new float[_g.nxyz()]{};
}

void Depletion::updateB10Abundance(const float& b10ap)
{
	_b10ap = b10ap;
	_b10fac = _b10ap / (_b10ap * B10AW + (100. - _b10ap) * B11AW);
	_b10wp = 100. * B10AW * _b10fac;
}


void Depletion::dep(const float& tsec)
{
    for (int l = 0; l < _g.nxyz(); ++l) {
        dep(l, tsec, _dnst, _dnst_new, _dnst_avg);
    }
}

void Depletion::dep(const int& l, const float& tsec, float* ati, float* atd, float* atavg)
{
    if(ati(U235,l) == 0) return;

    deph(l, tsec, ati, atd, atavg);
	for (int ihvy = 0; ihvy < NHEAVY; ihvy++)
	{
		ati(ISOHVY[ihvy], l) = atd(ISOHVY[ihvy], l);
	}
    depp(l, tsec, ati, atd,  atavg);
	ati(POIS, l) = atd(POIS, l);

	if (ism == SMType::SM_TR) {
		depsm(l, tsec, ati, atd, atavg);

		ati(PM47, l) = atd(PM47, l);
		ati(PS48, l) = atd(PS48, l);
		ati(PM48, l) = atd(PM48, l);
		ati(PM49, l) = atd(PM49, l);
		ati(SM49, l) = atd(SM49, l);
	}

	if (ixe == XEType::XE_TR) {
		depxe(l, tsec, ati, atd, atavg);
		ati(I135, l) = atd(I135, l);
		ati(XE45, l) = atd(XE45, l);
	}


}

void Depletion::deph(const int& l, const float& tsec, const float* ati, float* atd, float* atavg)
{
	for (int ic = 0; ic < _nhvychn; ++ic) {
		for (int i = 0; i < nheavy(ic); ++i) {
			atd(ihchn(i, ic), l) = 0.0;
			atavg(ihchn(i, ic), l) = 0.0;
		}
	}

	float r[10], exg[10], a[10][10];

	for (int ic = 0; ic < _nhvychn; ++ic) {

		for (int i = 0; i < nheavy(ic); ++i) {
			r[i] = 0.0;
			for (int j = 1; j < nheavy(ic); ++j) {
				a[i][j] = 0.0;
			}
		}

		for (int i = 0; i < nheavy(ic); ++i) {
			r[i] = rem(ihchn(i, ic), l);
			exg[i] = exp(-r[i] * tsec);

			int im1 = i - 1;

			if (i != 0) {
				for (int j = 0; j <= im1; ++j) {
					float gm1 = 0.0;

					switch (iptyp(i, ic)) {
					case(R_CAP):
						gm1 = cap(ihchn(im1, ic), l);
						if (ihchn(im1, ic) == Isotope::AM41) {
							if (ihchn(i, ic) == Isotope::CM42) {
								gm1 = gm1 * 0.71;
							}
							else if (ihchn(i, ic) == Isotope::AM42) {
								gm1 = gm1 * 0.14;
							}
							else if (ihchn(i, ic) == Isotope::PU42) {
								gm1 = gm1 * 0.15;
							}
						}
						break;
					case(R_DEC):
						gm1 = dcy(ihchn(im1, ic));
						break;
					case(R_N2N):
						gm1 = _tn2n[l];
						break;
					default:
						break;
					}

					a[i][j] = gm1 * a[im1][j] / (r[i] - r[j]);
				}
			}

			switch (idpct(i, ic)) {
			case (ChainAction::LEAVE):
				a[i][i] = 0.0;
				break;
			default:
				a[i][i] = ati(ihchn(i, ic), l);
				break;
			}

			if (i != 0) {
				for (int j = 0; j <= im1; ++j) {
					a[i][i] = a[i][i] - a[i][j];
				}
			}

			float dnew = 0.0;
			float ditg = 0.0;
			for (int j = 0; j <= i; ++j) {
				if (r[j] != 0.0) {
					dnew = dnew + a[i][j] * exg[j];
					ditg = ditg + a[i][j] * (1. - exg[j]) / r[j];
				}
			}

			if (idpct(i,ic) != ChainAction::HOLD) {
				atd(ihchn(i, ic), l) = atd(ihchn(i, ic), l) + dnew;
				atavg(ihchn(i, ic), l) = atavg(ihchn(i, ic), l) + ditg / tsec;
			}
		}
	}
}
void Depletion::depxe(const int& l, const float& tsec, const float* ati, float* atd, float* atavg)
{
	int ipp = I135;
	int idd = XE45;
	float remd = rem(I135, l);
	float dcp = dcy(XE45);

	float fyp = 0., fyd = 0.;
	for (int ih = 0; ih < NHEAVY; ++ih) {
		int ihf = ISOHVY[ih];
		fyp = fyp + fis(ihf, l) * fyld(IFP_I135, ih) * atavg(ihf, l);
		fyd = fyd + fis(ihf, l) * fyld(IFP_XE45, ih) * atavg(ihf, l);
	}
	float exgp = exp(-dcp * tsec);
	float exgd = exp(-remd * tsec);

	atd(ipp, l) = ati(ipp, l) * exgp + fyp / dcp * (1. - exgp);
	atd(idd, l) = ati(idd, l) * exgd + (fyp + fyd) / remd * (1. - exgd) + (ati(ipp, l) * dcp - fyp) / (remd - dcp) * (exgp - exgd);
}

void Depletion::eqxe(const float* xsmica, const float* xsmicf, const SOL_VAR* flux, const float& fnorm)
{

	if (ixe != XEType::XE_EQ) return;

	for (int l = 0; l < _g.nxyz(); l++)
	{
	    if(xsmicf(1,U235,l) == 0) continue;

		eqxe(l, xsmica, xsmicf, flux, fnorm);
	}
}


void Depletion::eqxe(const int& l, const float* xsmica, const float* xsmicf, const SOL_VAR* flux, const float& fnorm)
{

	float rem_i = 0.0;
	float rem_xe = 0.0;

	for (int ig = 0; ig < _g.ng(); ++ig) {
		rem_xe = rem_xe + xsmica(ig, XE45, l) * flux(ig, l);
	}
    rem_xe = rem_xe * fnorm * 1.E-24 + dcy(XE45);

	float fy_i135 = 0., fy_xe45 = 0.;
	for (int i = 0; i < NHEAVY; ++i) {
		int iso = ISOHVY[i];

		if (dnst(iso, l) == 0.0) continue;

		float fis = 0.0;
		for (int ig = 0; ig < _g.ng(); ++ig) {
			fis = fis + xsmicf(ig, iso, l) * flux(ig, l);
		}

        fy_i135 = fy_i135 + fis * fyld(IFP_I135, i) * dnst(iso, l);
        fy_xe45 = fy_xe45 + fis * fyld(IFP_XE45, i) * dnst(iso, l);
	}
    fy_i135 *= fnorm* 1.E-24;
    fy_xe45 *= fnorm* 1.E-24;


	dnst(I135, l) = fy_i135/ dcy(I135);
	dnst(XE45, l) = (fy_i135 + fy_xe45) / rem_xe;
}


void Depletion::depsm(const int& l, const float& tsec, const float* ati, float* atd, float* atavg)
{

	if (ism != SMType::SM_TR) return;

	float exp47 = exp(-rem(PM47, l) * tsec);
	float exp48s = exp(-rem(PS48, l) * tsec);
	float exp48m = exp(-rem(PM48, l) * tsec);
	float exp49 = exp(-rem(PM49, l) * tsec);
	float expsm = exp(-rem(SM49, l) * tsec);

	float fr47 = 0., fr49 = 0.;
	for (int ifiso = 0; ifiso < NHEAVY; ++ifiso) {
		int ihf = ISOHVY[ifiso];
		fr47 = fr47 + fis(ihf, l) * fyld(IFP_PM47, ifiso) * atavg(ihf, l);
		fr49 = fr49 + fis(ihf, l) * fyld(IFP_PM49, ifiso) * atavg(ihf, l);
	}


	// pm147
	float abs47 = rem(PM47, l) - dcy(PM47);
	float a47 = fr47 / rem(PM47, l);
	atd(PM47, l) = a47 * (1.0 - exp47) + ati(PM47, l) * exp47;

	// pm148 and pm148m
	float abs48s = rem(PS48, l) - dcy(PS48);
	float abs48m = rem(PM48, l) - dcy(PM48);

	float a48 = 0;
	float b48 = 0;
	float a4m = 0;
	float b4m = 0;

	if (abs47 != 0) {
		a48 = FRAC48 * abs47 * a47 / rem(PS48, l);
		b48 = FRAC48 * abs47 * (ati(PM47, l) - a47) / (rem(PS48, l) - rem(PM47, l));
		a4m = (1.0 - FRAC48) * abs47 * a47 / rem(PM48, l);
		b4m = (1.0 - FRAC48) * abs47 * (ati(PM47, l) - a47) / (rem(PM48, l) - rem(PM47, l));
	}

	atd(PS48, l) = a48 * (1.0 - exp48s) + b48 * (exp47 - exp48s) + ati(PS48, l) * exp48s;
	atd(PM48, l) = a4m * (1.0 - exp48m) + b4m * (exp47 - exp48m) + ati(PM48, l) * exp48m;

	// pm149
	float a49 = (fr49 + abs48s * a48 + abs48m * a4m) / rem(PM49, l);
	float b49 = 0;
	if (abs48s != 0) {
		b49 = abs48s * (ati(PS48, l) - a48 - b48) / (rem(PM49, l) - rem(PS48, l));
	}

	float c49 = 0;
	if (abs48m != 0) {
		c49 = abs48m * (ati(PM48, l) - a4m - b4m) / (rem(PM49, l) - rem(PM48, l));
	}

	float d49 = 0;
	if (abs47 != 0) {
		d49 = (abs48s * b48 + abs48m * b4m) / (rem(PM49, l) - rem(PM47, l));
	}

	atd(PM49,l) = a49 * (1.0 - exp49) + b49 * (exp48s - exp49) + c49 * (exp48m - exp49) + d49 * (exp47 - exp49) + ati(PM49, l) * exp49;

	// sm149
	float asm1 = dcy(PM49) * a49;
	float csm = dcy(PM49) * b49;
	float dsm = dcy(PM49) * c49;
	float esm = dcy(PM49) * d49;
	float bsm = dcy(PM49) * ati(PM49, l) - asm1 - csm - dsm - esm;

	atd(SM49, l) = asm1 / rem(SM49, l) * (1. - expsm)
		+ bsm / (rem(SM49, l) - rem(PM49, l)) * (exp49 - expsm)
		+ csm / (rem(SM49, l) - rem(PS48, l)) * (exp48s - expsm)
		+ dsm / (rem(SM49, l) - rem(PM48, l)) * (exp48m - expsm)
		+ esm / (rem(SM49, l) - rem(PM47, l)) * (exp47 - expsm)
		+ ati(SM49, l) * expsm;
}

void Depletion::depp(const int& l, const float& tsec, const float* ati, float* atd, float* atavg)
{
	atd(POIS,l) = ati(POIS,l) * exp(-cap(POIS,l) * tsec);
}


void Depletion::pickData(const float* xsmica, const float* xsmicf, const float* xsmic2n, const SOL_VAR* flux, const float& fnorm) {

    for (int l = 0; l < _g.nxyz(); ++l) {
		if (xsmica(1,U235, l) == 0) continue;

        pickData(l, xsmica, xsmicf, xsmic2n, flux, fnorm);
    }
}

void Depletion::pickData(const int& l, const float* xsmica, const float* xsmicf, const float* xsmic2n, const SOL_VAR* flux, const float& fnorm)
{
    //   calculate capture, removal, decay and fission rate of nuclides
	for (int iiso = 0; iiso < NDEP; ++iiso) {
		cap(iiso, l) = 0.0;
		rem(iiso, l) = 0.0;
		fis(iiso, l) = 0.0;

		for (int ig = 0; ig < _g.ng(); ++ig) {
            float xsa= xsmica(ig, iiso, l);
            float xsc= xsmica(ig, iiso, l) - xsmicf(ig, iiso, l);
            if (iiso == U238 && ig == 0) {
                xsa +=  2 * xsmic2n(ig, l);
                xsc +=  xsmic2n(ig, l);
            }

            cap(iiso, l) = cap(iiso, l) + xsc * flux(ig, l);
			rem(iiso, l) = rem(iiso, l) + xsa * flux(ig, l);
			fis(iiso, l) = fis(iiso, l) + xsmicf(ig, iiso, l) * flux(ig, l);
		}
		cap(iiso, l) = cap(iiso, l) * 1.0E-24*fnorm;
		rem(iiso, l) = rem(iiso, l) * 1.0E-24*fnorm + dcy(iiso);
		fis(iiso, l) = fis(iiso, l) * 1.0E-24*fnorm;
	}

	//   (n,2n) reaction rate
	_tn2n[l] = xsmic2n(0, l) * flux(0, l)* fnorm * 1.0E-24;
}

void Depletion::updateH2ODensity(const int& l, const float* dm, const float& ppm)
{
	dnst(H2O, l) = h2on(l) * dm[l];
	dnst(SB10, l) = 1.0E-06 * ppm * H2OAW * dnst(H2O, l) * _b10fac;

}

void Depletion::updateH2ODensity(const float* dm, const float& ppm) {
    for (int l = 0; l < _g.nxyz(); ++l) {
        updateH2ODensity(l, dm, ppm);
    }
}


