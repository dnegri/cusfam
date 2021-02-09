#pragma once
#include "pch.h"

#ifndef XS_PRECISION
#define XS_PRECISION float
#endif

class CrossSection : public Managed {
protected:
	int _ng;
	int _nxyz;

	XS_PRECISION* _xsnf;
	XS_PRECISION* _xsdf;
	XS_PRECISION* _xssf;
	XS_PRECISION* _xstf;
	XS_PRECISION* _xskf;
	XS_PRECISION* _chif;
	XS_PRECISION* _xsadf;

	//
	// pointers for micro cross section derivative for isotopes
	//
	// xd*micd    transport
	// xd*mica    absorption
	// xd*mics    scatter
	// xd*micf    fission
	// xd*mick    kappa-fission
	// xd*micn    nue-fission
	//
	// * = p for ppm derivatives
	//   = m for moderator temperature derivatives
	//   = d for moderator density derivatives
	//   = f for fuel temperature derivatives
	//
	//
	// (1) = ipm49th
	// (2) = isamth
	// (3) = ii135th
	// (4) = ixenth
	// (5) = ib10th
	// (6) = ih2oth
	//
	// xdpmicf = pointers for micro xs ppm derivatives for fission nuclides (nt2d,nz,ng,nfcnt)
	// xdfmicf = pointers for micro xs tf  derivatives for fission nuclides (nt2d,nz,ng,nfcnt)
	// xdmmicf = pointers for micro xs tm  derivatives for fission nuclides (nt2d,nz,ng,nfcnt)
	// xddmicf = pointers for micro xs dm  derivatives for fission nuclides (nt2d,nz,ng,nfcnt)
	//
	//
	XS_PRECISION* _xdpmicf;   // (:,:,:,:,:)
	XS_PRECISION* _xdfmicf;   // (:,:,:,:,:)
	XS_PRECISION* _xdmmicf;   // (:,:,:,:,:)
	XS_PRECISION* _xddmicf;   // (:,:,:,:,:)
	XS_PRECISION* _xdfmics;   // (:,:,:,:,:,:)
	XS_PRECISION* _xdfmica;   // (:,:,:,:,:)
	XS_PRECISION* _xdfmicd;   // (:,:,:,:,:)
	XS_PRECISION* _xdmmics;   // (:,:,:,:,:,:)
	XS_PRECISION* _xdmmica;   // (:,:,:,:,:)
	XS_PRECISION* _xdmmicd;   // (:,:,:,:,:)
	XS_PRECISION* _xddmics;   // (:,:,:,:,:,:)
	XS_PRECISION* _xddmica;   // (:,:,:,:,:)
	XS_PRECISION* _xddmicd;   // (:,:,:,:,:)
	XS_PRECISION* _xdpmics;   // (:,:,:,:,:,:)
	XS_PRECISION* _xdpmica;   // (:,:,:,:,:)
	XS_PRECISION* _xdpmicd;   // (:,:,:,:,:)
	XS_PRECISION* _xdpmicn;   // (:,:,:,:,:)
	XS_PRECISION* _xdfmicn;   // (:,:,:,:,:)
	XS_PRECISION* _xdmmicn;   // (:,:,:,:,:)
	XS_PRECISION* _xddmicn;   // (:,:,:,:,:)
	XS_PRECISION* _xdpmick;   // (:,:,:,:,:)
	XS_PRECISION* _xdfmick;   // (:,:,:,:,:)
	XS_PRECISION* _xdmmick;   // (:,:,:,:,:)
	XS_PRECISION* _xddmick;   // (:,:,:,:,:)

//
// pointers for micro cross sections
//
// xsmicd    transport
// xsmica    absorption
// xsmics    scatter
// xsmicf    fission
// xsmick    kappa-fission
// xsmicn    nue-fission
//
//        => those at intermediate stage
//           and at end of each burnup step(final)
//
// xsmicd0   transport
// xsmica0   absorption
// xsmics0   scatter
// xsmicf0   fission
// xsmick0   kappa-fission
// xsmicn0   nue-fission
//
//        => those at each burnup step
//
//
	XS_PRECISION* _xsmicd; // (:,:,:,:)
	XS_PRECISION* _xsmica; // (:,:,:,:)
	XS_PRECISION* _xsmics; // (:,:,:,:,:)
	XS_PRECISION* _xsmicf; // (:,:,:,:)
	XS_PRECISION* _xsmick; // (:,:,:,:)
	XS_PRECISION* _xsmicn; // (:,:,:,:)
	XS_PRECISION* _xsmic2n; // (:,:)

	XS_PRECISION* _xsmicd0; // (:,:,:,:)
	XS_PRECISION* _xsmica0; // (:,:,:,:)
	XS_PRECISION* _xsmics0; // (:,:,:,:,:)
	XS_PRECISION* _xsmicf0; // (:,:,:,:)
	XS_PRECISION* _xsmick0; // (:,:,:,:)
	XS_PRECISION* _xsmicn0; // (:,:,:,:)

	XS_PRECISION* _xsmacd0; // (:,:,:,:)
	XS_PRECISION* _xsmaca0; // (:,:,:,:)
	XS_PRECISION* _xsmacs0; // (:,:,:,:,:)
	XS_PRECISION* _xsmacf0; // (:,:,:,:)
	XS_PRECISION* _xsmack0; // (:,:,:,:)
	XS_PRECISION* _xsmacn0; // (:,:,:,:)

	XS_PRECISION* _dpmacd;
	XS_PRECISION* _dpmaca;
	XS_PRECISION* _dpmacf;
	XS_PRECISION* _dpmack;
	XS_PRECISION* _dpmacn;
	XS_PRECISION* _dpmacs;
	XS_PRECISION* _ddmacd;
	XS_PRECISION* _ddmaca;
	XS_PRECISION* _ddmacf;
	XS_PRECISION* _ddmack;
	XS_PRECISION* _ddmacn;
	XS_PRECISION* _ddmacs;
	XS_PRECISION* _dmmacd;
	XS_PRECISION* _dmmaca;
	XS_PRECISION* _dmmacf;
	XS_PRECISION* _dmmack;
	XS_PRECISION* _dmmacn;
	XS_PRECISION* _dmmacs;
	XS_PRECISION* _dfmacd;
	XS_PRECISION* _dfmaca;
	XS_PRECISION* _dfmacf;
	XS_PRECISION* _dfmack;
	XS_PRECISION* _dfmacn;
	XS_PRECISION* _dfmacs;


public:
	__host__ __device__ CrossSection() {
	};

	__host__ __device__ CrossSection(const int& ng, const int& nxyz) {
		init(ng, nxyz);
	}

	__host__ __device__ void init(const int& ng, const int& nxyz) {
		_ng = ng;
		_nxyz = nxyz;

		_xsnf = new XS_PRECISION[_ng * _nxyz]{};
		_xsdf = new XS_PRECISION[_ng * _nxyz]{};
		_xstf = new XS_PRECISION[_ng * _nxyz]{};
		_xskf = new XS_PRECISION[_ng * _nxyz]{};
		_chif = new XS_PRECISION[_ng * _nxyz]{};
		_xssf = new XS_PRECISION[_ng * _ng * _nxyz]{};
		_xsadf = new XS_PRECISION[_ng * _nxyz]{};

		_xsmacd0 = new XS_PRECISION[_ng * _nxyz]; // (:,:,:,:)
		_xsmaca0 = new XS_PRECISION[_ng * _nxyz]; // (:,:,:,:)
		_xsmacs0 = new XS_PRECISION[_ng * _ng * _nxyz]; // (:,:,:,:,:)
		_xsmacf0 = new XS_PRECISION[_ng * _nxyz]; // (:,:,:,:)
		_xsmack0 = new XS_PRECISION[_ng * _nxyz]; // (:,:,:,:)
		_xsmacn0 = new XS_PRECISION[_ng * _nxyz]; // (:,:,:,:)


		_xsmicd = new XS_PRECISION[_ng * NISO * _nxyz]; // (:,:,:,:)
		_xsmica = new XS_PRECISION[_ng * NISO * _nxyz]; // (:,:,:,:)
		_xsmics = new XS_PRECISION[_ng * _ng * NISO * _nxyz]; // (:,:,:,:,:)
		_xsmicf = new XS_PRECISION[_ng * NISO * _nxyz]; // (:,:,:,:)
		_xsmick = new XS_PRECISION[_ng * NISO * _nxyz]; // (:,:,:,:)
		_xsmicn = new XS_PRECISION[_ng * NISO * _nxyz]; // (:,:,:,:)

		_xsmic2n = new XS_PRECISION[_ng * NISO * _nxyz]; // (:,:)
		_xsmicd0 = new XS_PRECISION[_ng * NISO * _nxyz]; // (:,:,:,:)
		_xsmica0 = new XS_PRECISION[_ng * NISO * _nxyz]; // (:,:,:,:)
		_xsmics0 = new XS_PRECISION[_ng * _ng * NISO * _nxyz]; // (:,:,:,:,:)
		_xsmicf0 = new XS_PRECISION[_ng * NISO * _nxyz]; // (:,:,:,:)
		_xsmick0 = new XS_PRECISION[_ng * NISO * _nxyz]; // (:,:,:,:)
		_xsmicn0 = new XS_PRECISION[_ng * NISO * _nxyz]; // (:,:,:,:)

		_xdfmicd = new XS_PRECISION[_ng * NISO * _nxyz];
		_xdfmica = new XS_PRECISION[_ng * NISO * _nxyz];
		_xddmicd = new XS_PRECISION[_ng * NISO * _nxyz];
		_xddmica = new XS_PRECISION[_ng * NISO * _nxyz];
		_xdpmicd = new XS_PRECISION[_ng * NISO * _nxyz];
		_xdpmica = new XS_PRECISION[_ng * NISO * _nxyz];
		_xdmmicd = new XS_PRECISION[_ng * NPTM * NISO * _nxyz];
		_xdmmica = new XS_PRECISION[_ng * NPTM * NISO * _nxyz];
		_xddmics = new XS_PRECISION[_ng * _ng * NISO * _nxyz];
		_xdpmics = new XS_PRECISION[_ng * _ng * NISO * _nxyz];
		_xdfmics = new XS_PRECISION[_ng * _ng * NISO * _nxyz];
		_xdmmics = new XS_PRECISION[_ng * _ng * NPTM * NISO * _nxyz];

		_xdfmicn = new XS_PRECISION[_ng * NISO * _nxyz];
		_xddmicn = new XS_PRECISION[_ng * NISO * _nxyz];
		_xdpmicn = new XS_PRECISION[_ng * NISO * _nxyz];
		_xdmmicn = new XS_PRECISION[_ng * NPTM * NISO * _nxyz];
		_xdpmicf = new XS_PRECISION[_ng * NISO * _nxyz];
		_xdfmicf = new XS_PRECISION[_ng * NISO * _nxyz];
		_xddmicf = new XS_PRECISION[_ng * NISO * _nxyz];
		_xdmmicf = new XS_PRECISION[_ng * NPTM * NISO * _nxyz];
		_xdpmick = new XS_PRECISION[_ng * NISO * _nxyz];
		_xdfmick = new XS_PRECISION[_ng * NISO * _nxyz];
		_xddmick = new XS_PRECISION[_ng * NISO * _nxyz];
		_xdmmick = new XS_PRECISION[_ng * NPTM * NISO * _nxyz];


		_dpmacd = new XS_PRECISION[_nxyz * _ng];
		_dpmaca = new XS_PRECISION[_nxyz * _ng];
		_dpmacf = new XS_PRECISION[_nxyz * _ng];
		_dpmack = new XS_PRECISION[_nxyz * _ng];
		_dpmacn = new XS_PRECISION[_nxyz * _ng];
		_dpmacs = new XS_PRECISION[_nxyz * _ng * _ng];
		_ddmacd = new XS_PRECISION[_nxyz * _ng];
		_ddmaca = new XS_PRECISION[_nxyz * _ng];
		_ddmacf = new XS_PRECISION[_nxyz * _ng];
		_ddmack = new XS_PRECISION[_nxyz * _ng];
		_ddmacn = new XS_PRECISION[_nxyz * _ng];
		_ddmacs = new XS_PRECISION[_nxyz * _ng * _ng];
		_dmmacd = new XS_PRECISION[_nxyz * NPTM * _ng];
		_dmmaca = new XS_PRECISION[_nxyz * NPTM * _ng];
		_dmmacf = new XS_PRECISION[_nxyz * NPTM * _ng];
		_dmmack = new XS_PRECISION[_nxyz * NPTM * _ng];
		_dmmacn = new XS_PRECISION[_nxyz * NPTM * _ng];
		_dmmacs = new XS_PRECISION[_nxyz * NPTM * _ng * _ng];
		_dfmacd = new XS_PRECISION[_nxyz * _ng];
		_dfmaca = new XS_PRECISION[_nxyz * _ng];
		_dfmacf = new XS_PRECISION[_nxyz * _ng];
		_dfmack = new XS_PRECISION[_nxyz * _ng];
		_dfmacn = new XS_PRECISION[_nxyz * _ng];
		_dfmacs = new XS_PRECISION[_nxyz * _ng * _ng];


		for (size_t l = 0; l < _nxyz; l++)
		{
			chif(0, l) = 1.0;
			xsadf(0, l) = 1.0;
			xsadf(1, l) = 1.0;
		}

	};

	__host__ virtual ~CrossSection() {
		delete[] _xsnf;
		delete[] _xsdf;
		delete[] _xstf;
		delete[] _xskf;
		delete[] _chif;
		delete[] _xssf;

		delete[] _xsmicd;
		delete[] _xsmica;
		delete[] _xsmics;
		delete[] _xsmicf;
		delete[] _xsmick;
		delete[] _xsmicn;

		delete[] _xsmic2n;
		delete[] _xsmicd0;
		delete[] _xsmica0;
		delete[] _xsmics0;
		delete[] _xsmicf0;
		delete[] _xsmick0;
		delete[] _xsmicn0;

		delete[] _xdfmicd;
		delete[] _xdmmicd;
		delete[] _xddmicd;
		delete[] _xdpmicd;
		delete[] _xdpmicf;
		delete[] _xdfmicf;
		delete[] _xdmmicf;
		delete[] _xddmicf;
		delete[] _xdfmica;
		delete[] _xdmmica;
		delete[] _xddmica;
		delete[] _xdpmica;
		delete[] _xdpmicn;
		delete[] _xdfmicn;
		delete[] _xdmmicn;
		delete[] _xddmicn;

		delete[] _xdfmics;
		delete[] _xdmmics;
		delete[] _xddmics;
		delete[] _xdpmics;
	}


	__host__ __device__ const int& ng() const { return _ng; }
	__host__ __device__ const int& nxyz() const { return _nxyz; }

	__host__ __device__ void updateMacroXS(const int& l, float* dnst);
	__host__ __device__ void updateMacroXS(float* dnst);

	__host__ __device__ void updateXS(const int& l, const float* dnst, const float& dppm, const float& dtf, const float& dtm);
	void updateXS(const float* dnst, const float* dppm, const float* dtf, const float* dtm);


	__host__ __device__ inline XS_PRECISION& xsnf(const int& ig, const int& l) { return _xsnf[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsdf(const int& ig, const int& l) { return _xsdf[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xssf(const int& igs, const int& ige, const int& l) { return _xssf[l * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_PRECISION& xstf(const int& ig, const int& l) { return _xstf[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xskf(const int& ig, const int& l) { return _xskf[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& chif(const int& ig, const int& l) { return _chif[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsadf(const int& ig, const int& l) { return _xsadf[l * _ng + ig]; };

	__host__ __device__ inline XS_PRECISION& xdfmicd(const int& ig, const int& iiso, const int& l) { return _xdfmicd[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xddmicd(const int& ig, const int& iiso, const int& l) { return _xddmicd[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdpmicd(const int& ig, const int& iiso, const int& l) { return _xdpmicd[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdpmicf(const int& ig, const int& iiso, const int& l) { return _xdpmicf[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdfmicf(const int& ig, const int& iiso, const int& l) { return _xdfmicf[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xddmicf(const int& ig, const int& iiso, const int& l) { return _xddmicf[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdfmica(const int& ig, const int& iiso, const int& l) { return _xdfmica[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xddmica(const int& ig, const int& iiso, const int& l) { return _xddmica[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdpmica(const int& ig, const int& iiso, const int& l) { return _xdpmica[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdpmicn(const int& ig, const int& iiso, const int& l) { return _xdpmicn[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdfmicn(const int& ig, const int& iiso, const int& l) { return _xdfmicn[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xddmicn(const int& ig, const int& iiso, const int& l) { return _xddmicn[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdpmick(const int& ig, const int& iiso, const int& l) { return _xdpmick[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdfmick(const int& ig, const int& iiso, const int& l) { return _xdfmick[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xddmick(const int& ig, const int& iiso, const int& l) { return _xddmick[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdfmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xdfmics[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_PRECISION& xddmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xddmics[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_PRECISION& xdpmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xdpmics[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };

	__host__ __device__ inline XS_PRECISION& xdmmicd(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmicd[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdmmicf(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmicf[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdmmica(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmica[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdmmicn(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmicn[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdmmick(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmick[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xdmmics(const int& igs, const int& ige, const int& ip, const int& iiso, const int& l) { return _xdmmics[l * _ng * _ng * NPTM * NISO + iiso * _ng * _ng * NPTM + ip * _ng * _ng + ige * _ng + igs]; };

	__host__ __device__ inline XS_PRECISION& xsmic2n(const int& ig, const int& l) { return _xsmic2n[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmicd(const int& ig, const int& iiso, const int& l) { return _xsmicd[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmica(const int& ig, const int& iiso, const int& l) { return _xsmica[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xsmics[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_PRECISION& xsmicf(const int& ig, const int& iiso, const int& l) { return _xsmicf[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmick(const int& ig, const int& iiso, const int& l) { return _xsmick[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmicn(const int& ig, const int& iiso, const int& l) { return _xsmicn[l * _ng * NISO + iiso * _ng + ig]; };

	__host__ __device__ inline XS_PRECISION& xsmicd0(const int& ig, const int& iiso, const int& l) { return _xsmicd0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmica0(const int& ig, const int& iiso, const int& l) { return _xsmica0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmicf0(const int& ig, const int& iiso, const int& l) { return _xsmicf0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmick0(const int& ig, const int& iiso, const int& l) { return _xsmick0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmicn0(const int& ig, const int& iiso, const int& l) { return _xsmicn0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmics0(const int& igs, const int& ige, const int& iiso, const int& l) { return _xsmics0[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };


	__host__ __device__ inline XS_PRECISION& dpmacd(const int& ig, const int& l) { return _dpmacd[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dpmaca(const int& ig, const int& l) { return _dpmaca[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dpmacf(const int& ig, const int& l) { return _dpmacf[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dpmack(const int& ig, const int& l) { return _dpmack[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dpmacn(const int& ig, const int& l) { return _dpmacn[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dpmacs(const int& igs, const int& ige, const int& l) { return _dpmacs[l * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_PRECISION& ddmacd(const int& ig, const int& l) { return _ddmacd[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& ddmaca(const int& ig, const int& l) { return _ddmaca[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& ddmacf(const int& ig, const int& l) { return _ddmacf[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& ddmack(const int& ig, const int& l) { return _ddmack[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& ddmacn(const int& ig, const int& l) { return _ddmacn[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& ddmacs(const int& igs, const int& ige, const int& l) { return _ddmacs[l * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_PRECISION& dmmacd(const int& ig, const int& ip, const int& l) { return _dmmacd[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dmmaca(const int& ig, const int& ip, const int& l) { return _dmmaca[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dmmacf(const int& ig, const int& ip, const int& l) { return _dmmacf[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dmmack(const int& ig, const int& ip, const int& l) { return _dmmack[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dmmacn(const int& ig, const int& ip, const int& l) { return _dmmacn[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dmmacs(const int& igs, const int& ip, const int& ige, const int& l) { return _dmmacs[l * _ng * _ng * NPTM + ip * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_PRECISION& dfmacd(const int& ig, const int& l) { return _dfmacd[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dfmaca(const int& ig, const int& l) { return _dfmaca[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dfmacf(const int& ig, const int& l) { return _dfmacf[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dfmack(const int& ig, const int& l) { return _dfmack[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dfmacn(const int& ig, const int& l) { return _dfmacn[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& dfmacs(const int& igs, const int& ige, const int& l) { return _dfmacs[l * _ng * _ng + ige * _ng + igs]; };

	__host__ __device__ inline XS_PRECISION& xsmacd0(const int& ig, const int& l) { return _xsmacd0[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmaca0(const int& ig, const int& l) { return _xsmaca0[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmacf0(const int& ig, const int& l) { return _xsmacf0[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmack0(const int& ig, const int& l) { return _xsmack0[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmacn0(const int& ig, const int& l) { return _xsmacn0[l * _ng + ig]; };
	__host__ __device__ inline XS_PRECISION& xsmacs0(const int& igs, const int& ige, const int& l) { return _xsmacs0[l * _ng * _ng + ige * _ng + igs]; };

	const XS_PRECISION* xsnf() const { return _xsnf; };
	const XS_PRECISION* xsdf() const { return _xsdf; };
	const XS_PRECISION* xstf() const { return _xstf; };
	const XS_PRECISION* xskf() const { return _xskf; };
	const XS_PRECISION* chif() const { return _chif; };
	const XS_PRECISION* xssf() const { return _xssf; };
	const XS_PRECISION* xsadf() const { return _xsadf; };
	const XS_PRECISION* xsmacd0() const { return _xsmacd0; };
	const XS_PRECISION* xsmaca0() const { return _xsmaca0; };
	const XS_PRECISION* xsmacs0() const { return _xsmacs0; };
	const XS_PRECISION* xsmacf0() const { return _xsmacf0; };
	const XS_PRECISION* xsmack0() const { return _xsmack0; };
	const XS_PRECISION* xsmacn0() const { return _xsmacn0; };
	const XS_PRECISION* xsmicd() const { return _xsmicd; };
	const XS_PRECISION* xsmica() const { return _xsmica; };
	const XS_PRECISION* xsmics() const { return _xsmics; };
	const XS_PRECISION* xsmicf() const { return _xsmicf; };
	const XS_PRECISION* xsmick() const { return _xsmick; };
	const XS_PRECISION* xsmicn() const { return _xsmicn; };
	const XS_PRECISION* xsmic2n() const { return _xsmic2n; };
	const XS_PRECISION* xsmicd0() const { return _xsmicd0; };
	const XS_PRECISION* xsmica0() const { return _xsmica0; };
	const XS_PRECISION* xsmics0() const { return _xsmics0; };
	const XS_PRECISION* xsmicf0() const { return _xsmicf0; };
	const XS_PRECISION* xsmick0() const { return _xsmick0; };
	const XS_PRECISION* xsmicn0() const { return _xsmicn0; };
	const XS_PRECISION* xdfmicd() const { return _xdfmicd; };
	const XS_PRECISION* xdfmica() const { return _xdfmica; };
	const XS_PRECISION* xddmicd() const { return _xddmicd; };
	const XS_PRECISION* xddmica() const { return _xddmica; };
	const XS_PRECISION* xdpmicd() const { return _xdpmicd; };
	const XS_PRECISION* xdpmica() const { return _xdpmica; };
	const XS_PRECISION* xdmmicd() const { return _xdmmicd; };
	const XS_PRECISION* xdmmica() const { return _xdmmica; };
	const XS_PRECISION* xddmics() const { return _xddmics; };
	const XS_PRECISION* xdpmics() const { return _xdpmics; };
	const XS_PRECISION* xdfmics() const { return _xdfmics; };
	const XS_PRECISION* xdmmics() const { return _xdmmics; };
	const XS_PRECISION* xdfmicn() const { return _xdfmicn; };
	const XS_PRECISION* xddmicn() const { return _xddmicn; };
	const XS_PRECISION* xdpmicn() const { return _xdpmicn; };
	const XS_PRECISION* xdmmicn() const { return _xdmmicn; };
	const XS_PRECISION* xdpmicf() const { return _xdpmicf; };
	const XS_PRECISION* xdfmicf() const { return _xdfmicf; };
	const XS_PRECISION* xddmicf() const { return _xddmicf; };
	const XS_PRECISION* xdmmicf() const { return _xdmmicf; };
	const XS_PRECISION* xdpmick() const { return _xdpmick; };
	const XS_PRECISION* xdfmick() const { return _xdfmick; };
	const XS_PRECISION* xddmick() const { return _xddmick; };
	const XS_PRECISION* xdmmick() const { return _xdmmick; };
	const XS_PRECISION* dpmacd() const { return _dpmacd; };
	const XS_PRECISION* dpmaca() const { return _dpmaca; };
	const XS_PRECISION* dpmacf() const { return _dpmacf; };
	const XS_PRECISION* dpmack() const { return _dpmack; };
	const XS_PRECISION* dpmacn() const { return _dpmacn; };
	const XS_PRECISION* dpmacs() const { return _dpmacs; };
	const XS_PRECISION* ddmacd() const { return _ddmacd; };
	const XS_PRECISION* ddmaca() const { return _ddmaca; };
	const XS_PRECISION* ddmacf() const { return _ddmacf; };
	const XS_PRECISION* ddmack() const { return _ddmack; };
	const XS_PRECISION* ddmacn() const { return _ddmacn; };
	const XS_PRECISION* ddmacs() const { return _ddmacs; };
	const XS_PRECISION* dmmacd() const { return _dmmacd; };
	const XS_PRECISION* dmmaca() const { return _dmmaca; };
	const XS_PRECISION* dmmacf() const { return _dmmacf; };
	const XS_PRECISION* dmmack() const { return _dmmack; };
	const XS_PRECISION* dmmacn() const { return _dmmacn; };
	const XS_PRECISION* dmmacs() const { return _dmmacs; };
	const XS_PRECISION* dfmacd() const { return _dfmacd; };
	const XS_PRECISION* dfmaca() const { return _dfmaca; };
	const XS_PRECISION* dfmacf() const { return _dfmacf; };
	const XS_PRECISION* dfmack() const { return _dfmack; };
	const XS_PRECISION* dfmacn() const { return _dfmacn; };
	const XS_PRECISION* dfmacs() const { return _dfmacs; };


};
