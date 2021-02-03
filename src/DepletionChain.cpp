#include "DepletionChain.h"

#define xsmica(ig, iiso, l) xsmica[(l) * _g.ng() * _mnucl + (iiso) * _g.ng() + (ig)]
#define xsmicf(ig, iiso, l) xsmicf[(l) * _g.ng() * _mnucl + (iiso) * _g.ng() + (ig)]
#define xsmic2n(ig, l) xsmic2n[(l) * _g.ng() + (ig)]
#define phi(ig, l) phi[(l) * _g.ng() + (ig)]
#define ati(iiso, l) ati[(l) * _mnucl + (iiso)]
#define atd(iiso, l) atd[(l) * _mnucl + (iiso)]
#define atavg(iiso, l) atavg[(l) * _mnucl + (iiso)]


DepletionChain::DepletionChain(Geometry& g) : _g(g) {

	_mnucl = 25;
	_nfcnt = 12;

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

	_nfiso = 12;
	_nfpiso = 5;
	_fiso = new int[_nfiso] {U234, U235, U236, U238, NP37,
		NP39, PU48, PU49, PU40, PU41,
		PU42, AM43};

	_fpPM147 = 0;
	_fpPM149 = 1;
	_fpSM149 = 2;
	_fpI135 = 3;
	_fpXE145 = 4;

	_fpiso = new int[_nfpiso] {PM47, PM49, SM49, I135, XE45};
	_fyld = new float[_nfiso * _nfpiso]{
			2.017740E-02, 1.035690E-02, 0.0, 4.901130E-02, 6.763670E-03,
			2.246730E-02, 1.081620E-02, 0.0, 6.281870E-02, 2.566345E-03,
			2.295290E-02, 1.338370E-02, 0.0, 5.974780E-02, 1.049093E-03,
			2.592740E-02, 1.625290E-02, 0.0, 6.940720E-02, 2.686420E-04,
			2.500000E-02, 1.547160E-02, 0.0, 6.903040E-02, 7.720750E-03,
			2.500000E-02, 1.547160E-02, 0.0, 6.903040E-02, 7.720750E-03,
			2.236530E-02, 1.596690E-02, 0.0, 5.740170E-02, 9.935130E-03,
			2.002960E-02, 1.216300E-02, 0.0, 6.541880E-02, 1.066411E-02,
			2.123450E-02, 1.393890E-02, 0.0, 6.731600E-02, 5.001020E-03,
			2.284950E-02, 1.474070E-02, 0.0, 6.943130E-02, 2.269029E-03,
			2.387710E-02, 1.598400E-02, 0.0, 7.388510E-02, 1.057970E-03,
			2.336130E-02, 1.555480E-02, 0.0, 6.034700E-02, 7.250690E-03
	};

	_nsm = 5;
	_smids = new int[_nsm] {PM47, PS48, PM48, PM49, SM49};
	_nxe = 2;
	_xeids = new int[_nxe] {I135, XE45};


	_ndcy = 9;
	_dcyids = new int[_ndcy] {NP39, PU41, PU48, PM47, PS48, PM48, PM49, I135, XE45};
	_dcnst = new float[_ndcy] {3.40515E-06, 1.53705E-09, 2.50451E-10,
		8.37254E-09, 1.49451E-06, 1.94297E-07, 3.62737E-06,
		2.93061E-05, 2.10657E-05};


	float* _cap = new float[_mnucl * g.nxyz()];
	float* _rem = new float[_mnucl * g.nxyz()];
	float* _fis = new float[_mnucl * g.nxyz()];
	float* _dcy = new float[_mnucl * g.nxyz()];
	float* _tn2n = new float[g.nxyz()];

	ixe = XEType::XE_EQ;
	ism = SMType::SM_TR;

}

DepletionChain::~DepletionChain() {
	delete[] _nheavy;
	delete[] _hvyids;
	delete[] _reactype;
	delete[] _hvyupd;
	delete[] _fiso;
	delete[] _fpiso;
	delete[] _fyld;
	delete[] _smids;
	delete[] _xeids;
	delete[] _dcyids;
	delete[] _dcy;
}

__host__ __device__ void DepletionChain::dep(const int& l, const float& tsec, const float* ati, float* atd, float* atavg)
{

}

void DepletionChain::deph(const int& l, const float& tsec, const float* ati, float* atd, float* atavg)
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

			if (i == 0) continue;

			int im1 = i - 1;

			for (int j = 0; j < im1; ++j) {
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
					gm1 = dcy(ihchn(im1, ic), l);
					break;
				case(R_N2N):
					gm1 = _tn2n[l];
					break;
				default:
					break;
				}

				a[i][j] = gm1 * a[im1][j] / (r[i] - r[j]);
			}

			switch (idpct(i, ic)) {
			case (ChainAction::LEAVE):
				a[i][i] = 0.0;
				break;
			default:
				a[i][i] = ati(ihchn(i, ic), l);
			}

			if (i == 0) {
				for (int j = 0; j < im1 - 1; ++j) {
					a[i][i] = a[i][i] - a[i][j];
				}
			}

			float dnew = 0.0;
			float ditg = 0.0;
			for (int j = 1; j < i; ++j) {
				if (r[j] != 0.0) {
					dnew = dnew + a[i][j] * exg[j];
					ditg = ditg + a[i][j] * (1. - exg[j]) / r[j];
				}
			}

			if (idpct(ic, i) != ChainAction::HOLD) {
				atd(ihchn(i, ic), l) = atd(ihchn(i, ic), l) + dnew;
				atavg(ihchn(i, ic), l) = atavg(ihchn(i, ic), l) + ditg / tsec;
			}
		}
	}
}
void DepletionChain::depxe(const int& l, const float& tsec, const float* ati, float* atd, float* atavg)
{

	if (ixe != XEType::XE_TR) return;


	int ipp = I135;
	int idd = XE45;
	float remd = rem(I135, l);
	float dcp = dcy(XE45, l);

	float fyp = 0., fyd = 0.;
	for (int ih = 0; ih < _nfiso; ++ih) {
		int ihf = _fiso[ih];
		fyp = fyp + fis(ihf, l) * fyld(_fpI135, ih) * atavg(ihf, l);
		fyd = fyd + fis(ihf, l) * fyld(_fpXE145, ih) * atavg(ihf, l);
	}
	float exgp = exp(-dcp * tsec);
	float exgd = exp(-remd * tsec);

	atd(ipp, l) = ati(ipp, l) * exgp + fyp / dcp * (1. - exgp);
	atd(idd, l) = ati(idd, l) * exgd + (fyp + fyd) / remd * (1. - exgd) + (ati(ipp, l) * dcp - fyp) / (remd - dcp) * (exgp - exgd);
}


void DepletionChain::depsm(const int& l, const float& tsec, const float* ati, float* atd, float* atavg)
{

	if (ism != SMType::SM_TR) return;

	float exp47 = exp(-rem(PM47, l) * tsec);
	float exp48s = exp(-rem(PS48, l) * tsec);
	float exp48m = exp(-rem(PM48, l) * tsec);
	float exp49 = exp(-rem(PM49, l) * tsec);
	float expsm = exp(-rem(SM49, l) * tsec);

	float fr47 = 0., fr49 = 0.;
	for (int ih = 0; ih < _nfiso; ++ih) {
		int ihf = _fiso[ih];
		fr47 = fr47 + fis(ihf, l) * fyld(_fpI135, ih) * atavg(ihf, l);
		fr49 = fr49 + fis(ihf, l) * fyld(_fpXE145, ih) * atavg(ihf, l);
	}


	// pm147      
	float abs47 = rem(PM47, l) - dcy(PM47, l);
	float a47 = fr47 / rem(PM47, l);
	atd(PM47, l) = a47 * (1.0 - exp47) + ati(PM47, l) * exp47;

	// pm148 and pm148m
	float abs48s = rem(PS48, l) - dcy(PS48, l);
	float abs48m = rem(PM48, l) - dcy(PM48, l);

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
	float asm1 = dcy(PM49, l) * a49;
	float csm = dcy(PM49, l) * b49;
	float dsm = dcy(PM49, l) * c49;
	float esm = dcy(PM49, l) * d49;
	float bsm = dcy(PM49, l) * ati(PM49, l) - asm1 - csm - dsm - esm;

	atd(SM49, l) = asm1 / rem(SM49, l) * (1. - expsm)
		+ bsm / (rem(SM49, l) - rem(PM49, l)) * (exp49 - expsm)
		+ csm / (rem(SM49, l) - rem(PS48, l)) * (exp48s - expsm)
		+ dsm / (rem(SM49, l) - rem(PM48, l)) * (exp48m - expsm)
		+ esm / (rem(SM49, l) - rem(PM47, l)) * (exp47 - expsm)
		+ ati(SM49, l) * expsm;
}

void DepletionChain::depp(const int& l, const float& tsec, const float* ati, float* atd, float* atavg)
{
	atd(POIS,l) = ati(POIS,l) * exp(-cap(POIS,l) * tsec);
}

void DepletionChain::pickData(const int& l, const float* xsmica, const float* xsmicf, const float* xsmic2n, const double* phi)
{
	//for (int in = 1; in < _mnucl; ++in) {
		//        ati(in, l) = atom0(m2d,k,in)
		//        atd(in, l) = atom0(m2d,k,in)
		//}
		//   calculate capture, removal, decay and fission rate of nuclides
	for (int in = 1; in < _mnucl; ++in) {
		cap(in, l) = 0.0;
		rem(in, l) = 0.0;
		fis(in, l) = 0.0;

		for (int ig = 1; ig < _g.ng(); ++ig) {
			cap(in, l) = cap(in, l) + (xsmica(ig, in, l) - xsmicf(ig, in, l) + xsmic2n(ig, l)) * phi(ig, l);
			rem(in, l) = rem(in, l) + (xsmica(ig, in, l) + 2 * xsmic2n(ig, l)) * phi(ig, l);
			fis(in, l) = fis(in, l) + xsmicf(ig, in, l) * phi(ig, l);
		}
		cap(in, l) = cap(in, l) * 1.0E-24;
		rem(in, l) = rem(in, l) * 1.0E-24;
		fis(in, l) = fis(in, l) * 1.0E-24;
	}

	for (int in = 1; in < _ndcy; ++in) {
		int iiso = _dcyids[in];
		dcy(iiso, l) = _dcnst[in];
		rem(in, l) = rem(in, l) + dcy(iiso, l);
	}

	//   (n,2n) reaction rate
	_tn2n[l] = xsmic2n(0, l) * phi(0, l) * 1.0E-24;
}


