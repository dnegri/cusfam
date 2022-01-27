#pragma once
#include "pch.h"
#include "ControlRod.h"

class CrossSection : public Managed {
protected:
	int _ng;
	int _nxy;
	int _nxyz;

	XS_VAR* _xsnf;
	XS_VAR* _xsdf;
	XS_VAR* _xssf;
	XS_VAR* _xstf;
	XS_VAR* _xskf;
	XS_VAR* _chif;
	XS_VAR* _xsadf;

	XS_VAR* _xehfp;

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
	XS_VAR* _xdpmicf;   // (:,:,:,:,:)
	XS_VAR* _xdfmicf;   // (:,:,:,:,:)
	XS_VAR* _xdmmicf;   // (:,:,:,:,:)
	XS_VAR* _xddmicf;   // (:,:,:,:,:)
	XS_VAR* _xdfmics;   // (:,:,:,:,:,:)
	XS_VAR* _xdfmica;   // (:,:,:,:,:)
	XS_VAR* _xdfmicd;   // (:,:,:,:,:)
	XS_VAR* _xdmmics;   // (:,:,:,:,:,:)
	XS_VAR* _xdmmica;   // (:,:,:,:,:)
	XS_VAR* _xdmmicd;   // (:,:,:,:,:)
	XS_VAR* _xddmics;   // (:,:,:,:,:,:)
	XS_VAR* _xddmica;   // (:,:,:,:,:)
	XS_VAR* _xddmicd;   // (:,:,:,:,:)
	XS_VAR* _xdpmics;   // (:,:,:,:,:,:)
	XS_VAR* _xdpmica;   // (:,:,:,:,:)
	XS_VAR* _xdpmicd;   // (:,:,:,:,:)
	XS_VAR* _xdpmicn;   // (:,:,:,:,:)
	XS_VAR* _xdfmicn;   // (:,:,:,:,:)
	XS_VAR* _xdmmicn;   // (:,:,:,:,:)
	XS_VAR* _xddmicn;   // (:,:,:,:,:)
	XS_VAR* _xdpmick;   // (:,:,:,:,:)
	XS_VAR* _xdfmick;   // (:,:,:,:,:)
	XS_VAR* _xdmmick;   // (:,:,:,:,:)
	XS_VAR* _xddmick;   // (:,:,:,:,:)

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
	XS_VAR* _xsmicd; // (:,:,:,:)
	XS_VAR* _xsmica; // (:,:,:,:)
	XS_VAR* _xsmics; // (:,:,:,:,:)
	XS_VAR* _xsmicf; // (:,:,:,:)
	XS_VAR* _xsmick; // (:,:,:,:)
	XS_VAR* _xsmicn; // (:,:,:,:)
	XS_VAR* _xsmic2n; // (:,:)

	XS_VAR* _xsmicd0; // (:,:,:,:)
	XS_VAR* _xsmica0; // (:,:,:,:)
	XS_VAR* _xsmics0; // (:,:,:,:,:)
	XS_VAR* _xsmicf0; // (:,:,:,:)
	XS_VAR* _xsmick0; // (:,:,:,:)
	XS_VAR* _xsmicn0; // (:,:,:,:)

	XS_VAR* _xsmacd0; // (:,:,:,:)
	XS_VAR* _xsmaca0; // (:,:,:,:)
	XS_VAR* _xsmacs0; // (:,:,:,:,:)
	XS_VAR* _xsmacf0; // (:,:,:,:)
	XS_VAR* _xsmack0; // (:,:,:,:)
	XS_VAR* _xsmacn0; // (:,:,:,:)

	XS_VAR* _dpmacd;
	XS_VAR* _dpmaca;
	XS_VAR* _dpmacf;
	XS_VAR* _dpmack;
	XS_VAR* _dpmacn;
	XS_VAR* _dpmacs;
	XS_VAR* _ddmacd;
	XS_VAR* _ddmaca;
	XS_VAR* _ddmacf;
	XS_VAR* _ddmack;
	XS_VAR* _ddmacn;
	XS_VAR* _ddmacs;
	XS_VAR* _dmmacd;
	XS_VAR* _dmmaca;
	XS_VAR* _dmmacf;
	XS_VAR* _dmmack;
	XS_VAR* _dmmacn;
	XS_VAR* _dmmacs;
	XS_VAR* _dfmacd;
	XS_VAR* _dfmaca;
	XS_VAR* _dfmacf;
	XS_VAR* _dfmack;
	XS_VAR* _dfmacn;
	XS_VAR* _dfmacs;


public:
	__host__ __device__ CrossSection() {
	};

	__host__ __device__ CrossSection(const int& ng, const int& nxy, const int& nxyz) {
		init(ng, nxy, nxyz);
	}

	__host__ __device__ void init(const int& ng, const int& nxy, const int& nxyz) {
		_ng = ng;
		_nxyz = nxyz;
		_nxy = nxy;

		_xsnf = new XS_VAR[_ng * _nxyz]{};
		_xsdf = new XS_VAR[_ng * _nxyz]{};
		_xstf = new XS_VAR[_ng * _nxyz]{};
		_xskf = new XS_VAR[_ng * _nxyz]{};
		_chif = new XS_VAR[_ng * _nxyz]{};
		_xssf = new XS_VAR[_ng * _ng * _nxyz]{};
		_xsadf = new XS_VAR[_ng * _nxyz]{};
		_xehfp = new XS_VAR[ _nxyz]{};

		_xsmacd0 = new XS_VAR[_ng * _nxyz]; // (:,:,:,:)
		_xsmaca0 = new XS_VAR[_ng * _nxyz]; // (:,:,:,:)
		_xsmacs0 = new XS_VAR[_ng * _ng * _nxyz]; // (:,:,:,:,:)
		_xsmacf0 = new XS_VAR[_ng * _nxyz]; // (:,:,:,:)
		_xsmack0 = new XS_VAR[_ng * _nxyz]; // (:,:,:,:)
		_xsmacn0 = new XS_VAR[_ng * _nxyz]; // (:,:,:,:)


		_xsmicd = new XS_VAR[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmica = new XS_VAR[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmics = new XS_VAR[_ng * _ng * NISO * _nxyz]{}; // (:,:,:,:,:)
		_xsmicf = new XS_VAR[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmick = new XS_VAR[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmicn = new XS_VAR[_ng * NISO * _nxyz]{}; // (:,:,:,:)

		_xsmic2n = new XS_VAR[_ng * NISO * _nxyz]{}; // (:,:)
		_xsmicd0 = new XS_VAR[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmica0 = new XS_VAR[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmics0 = new XS_VAR[_ng * _ng * NISO * _nxyz]{}; // (:,:,:,:,:)
		_xsmicf0 = new XS_VAR[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmick0 = new XS_VAR[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmicn0 = new XS_VAR[_ng * NISO * _nxyz]{}; // (:,:,:,:)

		_xdfmicd = new XS_VAR[_ng * NISO * _nxyz]{};
		_xdfmica = new XS_VAR[_ng * NISO * _nxyz]{};
		_xddmicd = new XS_VAR[_ng * NISO * _nxyz]{};
		_xddmica = new XS_VAR[_ng * NISO * _nxyz]{};
		_xdpmicd = new XS_VAR[_ng * NISO * _nxyz]{};
		_xdpmica = new XS_VAR[_ng * NISO * _nxyz]{};
		_xdmmicd = new XS_VAR[_ng * NPTM * NISO * _nxyz]{};
		_xdmmica = new XS_VAR[_ng * NPTM * NISO * _nxyz]{};
		_xddmics = new XS_VAR[_ng * _ng * NISO * _nxyz]{};
		_xdpmics = new XS_VAR[_ng * _ng * NISO * _nxyz]{};
		_xdfmics = new XS_VAR[_ng * _ng * NISO * _nxyz]{};
		_xdmmics = new XS_VAR[_ng * _ng * NPTM * NISO * _nxyz]{};

		_xdfmicn = new XS_VAR[_ng * NISO * _nxyz]{};
		_xddmicn = new XS_VAR[_ng * NISO * _nxyz]{};
		_xdpmicn = new XS_VAR[_ng * NISO * _nxyz]{};
		_xdmmicn = new XS_VAR[_ng * NPTM * NISO * _nxyz]{};
		_xdpmicf = new XS_VAR[_ng * NISO * _nxyz]{};
		_xdfmicf = new XS_VAR[_ng * NISO * _nxyz]{};
		_xddmicf = new XS_VAR[_ng * NISO * _nxyz]{};
		_xdmmicf = new XS_VAR[_ng * NPTM * NISO * _nxyz]{};
		_xdpmick = new XS_VAR[_ng * NISO * _nxyz]{};
		_xdfmick = new XS_VAR[_ng * NISO * _nxyz]{};
		_xddmick = new XS_VAR[_ng * NISO * _nxyz]{};
		_xdmmick = new XS_VAR[_ng * NPTM * NISO * _nxyz]{};


		_dpmacd = new XS_VAR[_nxyz * _ng]{};
		_dpmaca = new XS_VAR[_nxyz * _ng]{};
		_dpmacf = new XS_VAR[_nxyz * _ng]{};
		_dpmack = new XS_VAR[_nxyz * _ng]{};
		_dpmacn = new XS_VAR[_nxyz * _ng]{};
		_dpmacs = new XS_VAR[_nxyz * _ng * _ng]{};
		_ddmacd = new XS_VAR[_nxyz * _ng]{};
		_ddmaca = new XS_VAR[_nxyz * _ng]{};
		_ddmacf = new XS_VAR[_nxyz * _ng]{};
		_ddmack = new XS_VAR[_nxyz * _ng]{};
		_ddmacn = new XS_VAR[_nxyz * _ng]{};
		_ddmacs = new XS_VAR[_nxyz * _ng * _ng]{};
		_dmmacd = new XS_VAR[_nxyz * NPTM * _ng]{};
		_dmmaca = new XS_VAR[_nxyz * NPTM * _ng]{};
		_dmmacf = new XS_VAR[_nxyz * NPTM * _ng]{};
		_dmmack = new XS_VAR[_nxyz * NPTM * _ng]{};
		_dmmacn = new XS_VAR[_nxyz * NPTM * _ng]{};
		_dmmacs = new XS_VAR[_nxyz * NPTM * _ng * _ng]{};
		_dfmacd = new XS_VAR[_nxyz * _ng]{};
		_dfmaca = new XS_VAR[_nxyz * _ng]{};
		_dfmacf = new XS_VAR[_nxyz * _ng]{};
		_dfmack = new XS_VAR[_nxyz * _ng]{};
		_dfmacn = new XS_VAR[_nxyz * _ng]{};
		_dfmacs = new XS_VAR[_nxyz * _ng * _ng]{};


		for (int l = 0; l < _nxyz; l++)
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
	__host__ void updateMacroXS(float* dnst);

	__host__ __device__ void updateXS(const int& l, const float* dnst, const float& dppm, const float& dtf, const float& dtm);
    __host__ void updateXS(const float* dnst, const float* dppm, const float* dtf, const float* dtm);

	__host__ __device__ void updateXenonXS(const int& l, const float& dppm, const float& dtf, const float& dtm);
	__host__ void updateXenonXS(const float* dppm, const float* dtf, const float* dtm);

	__host__ __device__ void updateXeSmXS(const int& l, const float& dppm, const float& dtf, const float& dtm);
	__host__ void updateXeSmXS(const float* dppm, const float* dtf, const float* dtm);

	__host__ __device__ void updateDepletionXS(const int& l, const float& dppm, const float& dtf, const float& dtm);
	__host__ void updateDepletionXS(const float* dppm, const float* dtf, const float* dtm);

	__host__ __device__ void updateRodXS(const int& l, const int& iso_rod, const float& ratio, const float& dppm, const float& dtf, const float& dtm);
    __host__ void updateRodXS(ControlRod& r, const float* dppm, const float* dtf, const float* dtm);

    __host__ __device__ inline XS_VAR& xsnf(const int& ig, const int& l) { return _xsnf[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsdf(const int& ig, const int& l) { return _xsdf[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xssf(const int& igs, const int& ige, const int& l) { return _xssf[l * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_VAR& xstf(const int& ig, const int& l) { return _xstf[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xskf(const int& ig, const int& l) { return _xskf[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& chif(const int& ig, const int& l) { return _chif[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsadf(const int& ig, const int& l) { return _xsadf[l * _ng + ig]; };

	__host__ __device__ inline XS_VAR& xdfmicd(const int& ig, const int& iiso, const int& l) { return _xdfmicd[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xddmicd(const int& ig, const int& iiso, const int& l) { return _xddmicd[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdpmicd(const int& ig, const int& iiso, const int& l) { return _xdpmicd[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdpmicf(const int& ig, const int& iiso, const int& l) { return _xdpmicf[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdfmicf(const int& ig, const int& iiso, const int& l) { return _xdfmicf[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xddmicf(const int& ig, const int& iiso, const int& l) { return _xddmicf[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdfmica(const int& ig, const int& iiso, const int& l) { return _xdfmica[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xddmica(const int& ig, const int& iiso, const int& l) { return _xddmica[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdpmica(const int& ig, const int& iiso, const int& l) { return _xdpmica[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdpmicn(const int& ig, const int& iiso, const int& l) { return _xdpmicn[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdfmicn(const int& ig, const int& iiso, const int& l) { return _xdfmicn[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xddmicn(const int& ig, const int& iiso, const int& l) { return _xddmicn[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdpmick(const int& ig, const int& iiso, const int& l) { return _xdpmick[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdfmick(const int& ig, const int& iiso, const int& l) { return _xdfmick[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xddmick(const int& ig, const int& iiso, const int& l) { return _xddmick[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdfmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xdfmics[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_VAR& xddmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xddmics[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_VAR& xdpmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xdpmics[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };

	__host__ __device__ inline XS_VAR& xdmmicd(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmicd[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdmmicf(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmicf[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdmmica(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmica[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdmmicn(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmicn[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdmmick(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmick[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xdmmics(const int& igs, const int& ige, const int& ip, const int& iiso, const int& l) { return _xdmmics[l * _ng * _ng * NPTM * NISO + iiso * _ng * _ng * NPTM + ip * _ng * _ng + ige * _ng + igs]; };

	__host__ __device__ inline XS_VAR& xsmic2n(const int& ig, const int& l) { return _xsmic2n[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmicd(const int& ig, const int& iiso, const int& l) { return _xsmicd[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmica(const int& ig, const int& iiso, const int& l) { return _xsmica[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xsmics[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_VAR& xsmicf(const int& ig, const int& iiso, const int& l) { return _xsmicf[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmick(const int& ig, const int& iiso, const int& l) { return _xsmick[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmicn(const int& ig, const int& iiso, const int& l) { return _xsmicn[l * _ng * NISO + iiso * _ng + ig]; };

	__host__ __device__ inline XS_VAR& xsmicd0(const int& ig, const int& iiso, const int& l) { return _xsmicd0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmica0(const int& ig, const int& iiso, const int& l) { return _xsmica0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmicf0(const int& ig, const int& iiso, const int& l) { return _xsmicf0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmick0(const int& ig, const int& iiso, const int& l) { return _xsmick0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmicn0(const int& ig, const int& iiso, const int& l) { return _xsmicn0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmics0(const int& igs, const int& ige, const int& iiso, const int& l) { return _xsmics0[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };


	__host__ __device__ inline XS_VAR& dpmacd(const int& ig, const int& l) { return _dpmacd[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dpmaca(const int& ig, const int& l) { return _dpmaca[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dpmacf(const int& ig, const int& l) { return _dpmacf[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dpmack(const int& ig, const int& l) { return _dpmack[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dpmacn(const int& ig, const int& l) { return _dpmacn[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dpmacs(const int& igs, const int& ige, const int& l) { return _dpmacs[l * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_VAR& ddmacd(const int& ig, const int& l) { return _ddmacd[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& ddmaca(const int& ig, const int& l) { return _ddmaca[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& ddmacf(const int& ig, const int& l) { return _ddmacf[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& ddmack(const int& ig, const int& l) { return _ddmack[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& ddmacn(const int& ig, const int& l) { return _ddmacn[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& ddmacs(const int& igs, const int& ige, const int& l) { return _ddmacs[l * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_VAR& dmmacd(const int& ig, const int& ip, const int& l) { return _dmmacd[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dmmaca(const int& ig, const int& ip, const int& l) { return _dmmaca[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dmmacf(const int& ig, const int& ip, const int& l) { return _dmmacf[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dmmack(const int& ig, const int& ip, const int& l) { return _dmmack[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dmmacn(const int& ig, const int& ip, const int& l) { return _dmmacn[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dmmacs(const int& igs, const int& ige, const int& ip, const int& l) { return _dmmacs[l * _ng * _ng * NPTM + ip * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline XS_VAR& dfmacd(const int& ig, const int& l) { return _dfmacd[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dfmaca(const int& ig, const int& l) { return _dfmaca[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dfmacf(const int& ig, const int& l) { return _dfmacf[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dfmack(const int& ig, const int& l) { return _dfmack[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dfmacn(const int& ig, const int& l) { return _dfmacn[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& dfmacs(const int& igs, const int& ige, const int& l) { return _dfmacs[l * _ng * _ng + ige * _ng + igs]; };

	__host__ __device__ inline XS_VAR& xsmacd0(const int& ig, const int& l) { return _xsmacd0[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmaca0(const int& ig, const int& l) { return _xsmaca0[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmacf0(const int& ig, const int& l) { return _xsmacf0[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmack0(const int& ig, const int& l) { return _xsmack0[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmacn0(const int& ig, const int& l) { return _xsmacn0[l * _ng + ig]; };
	__host__ __device__ inline XS_VAR& xsmacs0(const int& igs, const int& ige, const int& l) { return _xsmacs0[l * _ng * _ng + ige * _ng + igs]; };

	const XS_VAR* xsnf() const { return _xsnf; };
	const XS_VAR* xsdf() const { return _xsdf; };
	const XS_VAR* xstf() const { return _xstf; };
	const XS_VAR* xskf() const { return _xskf; };
	const XS_VAR* chif() const { return _chif; };
	const XS_VAR* xssf() const { return _xssf; };
	const XS_VAR* xsadf() const { return _xsadf; };
	const XS_VAR* xsmacd0() const { return _xsmacd0; };
	const XS_VAR* xsmaca0() const { return _xsmaca0; };
	const XS_VAR* xsmacs0() const { return _xsmacs0; };
	const XS_VAR* xsmacf0() const { return _xsmacf0; };
	const XS_VAR* xsmack0() const { return _xsmack0; };
	const XS_VAR* xsmacn0() const { return _xsmacn0; };
	const XS_VAR* xsmicd() const { return _xsmicd; };
	const XS_VAR* xsmica() const { return _xsmica; };
	const XS_VAR* xsmics() const { return _xsmics; };
	const XS_VAR* xsmicf() const { return _xsmicf; };
	const XS_VAR* xsmick() const { return _xsmick; };
	const XS_VAR* xsmicn() const { return _xsmicn; };
	const XS_VAR* xsmic2n() const { return _xsmic2n; };
	const XS_VAR* xsmicd0() const { return _xsmicd0; };
	const XS_VAR* xsmica0() const { return _xsmica0; };
	const XS_VAR* xsmics0() const { return _xsmics0; };
	const XS_VAR* xsmicf0() const { return _xsmicf0; };
	const XS_VAR* xsmick0() const { return _xsmick0; };
	const XS_VAR* xsmicn0() const { return _xsmicn0; };
	const XS_VAR* xdfmicd() const { return _xdfmicd; };
	const XS_VAR* xdfmica() const { return _xdfmica; };
	const XS_VAR* xddmicd() const { return _xddmicd; };
	const XS_VAR* xddmica() const { return _xddmica; };
	const XS_VAR* xdpmicd() const { return _xdpmicd; };
	const XS_VAR* xdpmica() const { return _xdpmica; };
	const XS_VAR* xdmmicd() const { return _xdmmicd; };
	const XS_VAR* xdmmica() const { return _xdmmica; };
	const XS_VAR* xddmics() const { return _xddmics; };
	const XS_VAR* xdpmics() const { return _xdpmics; };
	const XS_VAR* xdfmics() const { return _xdfmics; };
	const XS_VAR* xdmmics() const { return _xdmmics; };
	const XS_VAR* xdfmicn() const { return _xdfmicn; };
	const XS_VAR* xddmicn() const { return _xddmicn; };
	const XS_VAR* xdpmicn() const { return _xdpmicn; };
	const XS_VAR* xdmmicn() const { return _xdmmicn; };
	const XS_VAR* xdpmicf() const { return _xdpmicf; };
	const XS_VAR* xdfmicf() const { return _xdfmicf; };
	const XS_VAR* xddmicf() const { return _xddmicf; };
	const XS_VAR* xdmmicf() const { return _xdmmicf; };
	const XS_VAR* xdpmick() const { return _xdpmick; };
	const XS_VAR* xdfmick() const { return _xdfmick; };
	const XS_VAR* xddmick() const { return _xddmick; };
	const XS_VAR* xdmmick() const { return _xdmmick; };
	const XS_VAR* dpmacd() const { return _dpmacd; };
	const XS_VAR* dpmaca() const { return _dpmaca; };
	const XS_VAR* dpmacf() const { return _dpmacf; };
	const XS_VAR* dpmack() const { return _dpmack; };
	const XS_VAR* dpmacn() const { return _dpmacn; };
	const XS_VAR* dpmacs() const { return _dpmacs; };
	const XS_VAR* ddmacd() const { return _ddmacd; };
	const XS_VAR* ddmaca() const { return _ddmaca; };
	const XS_VAR* ddmacf() const { return _ddmacf; };
	const XS_VAR* ddmack() const { return _ddmack; };
	const XS_VAR* ddmacn() const { return _ddmacn; };
	const XS_VAR* ddmacs() const { return _ddmacs; };
	const XS_VAR* dmmacd() const { return _dmmacd; };
	const XS_VAR* dmmaca() const { return _dmmaca; };
	const XS_VAR* dmmacf() const { return _dmmacf; };
	const XS_VAR* dmmack() const { return _dmmack; };
	const XS_VAR* dmmacn() const { return _dmmacn; };
	const XS_VAR* dmmacs() const { return _dmmacs; };
	const XS_VAR* dfmacd() const { return _dfmacd; };
	const XS_VAR* dfmaca() const { return _dfmaca; };
	const XS_VAR* dfmacf() const { return _dfmacf; };
	const XS_VAR* dfmack() const { return _dfmack; };
	const XS_VAR* dfmacn() const { return _dfmacn; };
	const XS_VAR* dfmacs() const { return _dfmacs; };

	__host__ __device__ inline XS_VAR* xsnf(const int& l) { return &_xsnf[l * _ng]; };
	__host__ __device__ inline XS_VAR* xsdf(const int& l) { return &_xsdf[l * _ng]; };
	__host__ __device__ inline XS_VAR* xssf(const int& l) { return &_xssf[l * _ng * _ng]; };
	__host__ __device__ inline XS_VAR* xstf(const int& l) { return &_xstf[l * _ng]; };
	__host__ __device__ inline XS_VAR* xskf(const int& l) { return &_xskf[l * _ng]; };
	__host__ __device__ inline XS_VAR* chif(const int& l) { return &_chif[l * _ng]; };
	__host__ __device__ inline XS_VAR* xsadf(const int& l) { return &_xsadf[l * _ng]; };

	__host__ __device__ inline XS_VAR* xdfmicd(const int& iiso, const int& l) { return &_xdfmicd[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xddmicd(const int& iiso, const int& l) { return &_xddmicd[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xdpmicd(const int& iiso, const int& l) { return &_xdpmicd[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xdpmicf(const int& iiso, const int& l) { return &_xdpmicf[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xdfmicf(const int& iiso, const int& l) { return &_xdfmicf[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xddmicf(const int& iiso, const int& l) { return &_xddmicf[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xdfmica(const int& iiso, const int& l) { return &_xdfmica[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xddmica(const int& iiso, const int& l) { return &_xddmica[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xdpmica(const int& iiso, const int& l) { return &_xdpmica[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xdpmicn(const int& iiso, const int& l) { return &_xdpmicn[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xdfmicn(const int& iiso, const int& l) { return &_xdfmicn[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xddmicn(const int& iiso, const int& l) { return &_xddmicn[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xdpmick(const int& iiso, const int& l) { return &_xdpmick[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xdfmick(const int& iiso, const int& l) { return &_xdfmick[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xddmick(const int& iiso, const int& l) { return &_xddmick[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xdfmics(const int& iiso, const int& l) { return &_xdfmics[l * _ng * _ng * NISO + iiso * _ng * _ng]; };
	__host__ __device__ inline XS_VAR* xddmics(const int& iiso, const int& l) { return &_xddmics[l * _ng * _ng * NISO + iiso * _ng * _ng]; };
	__host__ __device__ inline XS_VAR* xdpmics(const int& iiso, const int& l) { return &_xdpmics[l * _ng * _ng * NISO + iiso * _ng * _ng]; };

	__host__ __device__ inline XS_VAR* xdmmicd(const int& ip, const int& iiso, const int& l) { return &_xdmmicd[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline XS_VAR* xdmmicf(const int& ip, const int& iiso, const int& l) { return &_xdmmicf[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline XS_VAR* xdmmica(const int& ip, const int& iiso, const int& l) { return &_xdmmica[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline XS_VAR* xdmmicn(const int& ip, const int& iiso, const int& l) { return &_xdmmicn[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline XS_VAR* xdmmick(const int& ip, const int& iiso, const int& l) { return &_xdmmick[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline XS_VAR* xdmmics(const int& ip, const int& iiso, const int& l) { return &_xdmmics[l * _ng * _ng * NPTM * NISO + iiso * _ng * _ng * NPTM + ip * _ng * _ng]; };

	__host__ __device__ inline XS_VAR* xsmic2n(const int& l) { return &_xsmic2n[l * _ng]; };
	__host__ __device__ inline XS_VAR* xsmicd(const int& iiso, const int& l) { return &_xsmicd[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xsmica(const int& iiso, const int& l) { return &_xsmica[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xsmics(const int& iiso, const int& l) { return &_xsmics[l * _ng * _ng * NISO + iiso * _ng * _ng]; };
	__host__ __device__ inline XS_VAR* xsmicf(const int& iiso, const int& l) { return &_xsmicf[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xsmick(const int& iiso, const int& l) { return &_xsmick[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xsmicn(const int& iiso, const int& l) { return &_xsmicn[l * _ng * NISO + iiso * _ng]; };

	__host__ __device__ inline XS_VAR* xsmicd0(const int& iiso, const int& l) { return &_xsmicd0[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xsmica0(const int& iiso, const int& l) { return &_xsmica0[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xsmicf0(const int& iiso, const int& l) { return &_xsmicf0[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xsmick0(const int& iiso, const int& l) { return &_xsmick0[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xsmicn0(const int& iiso, const int& l) { return &_xsmicn0[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline XS_VAR* xsmics0(const int& iiso, const int& l) { return &_xsmics0[l * _ng * _ng * NISO + iiso * _ng * _ng]; };


	__host__ __device__ inline XS_VAR* dpmacd(const int& l) { return &_dpmacd[l * _ng]; };
	__host__ __device__ inline XS_VAR* dpmaca(const int& l) { return &_dpmaca[l * _ng]; };
	__host__ __device__ inline XS_VAR* dpmacf(const int& l) { return &_dpmacf[l * _ng]; };
	__host__ __device__ inline XS_VAR* dpmack(const int& l) { return &_dpmack[l * _ng]; };
	__host__ __device__ inline XS_VAR* dpmacn(const int& l) { return &_dpmacn[l * _ng]; };
	__host__ __device__ inline XS_VAR* dpmacs(const int& l) { return &_dpmacs[l * _ng * _ng]; };
	__host__ __device__ inline XS_VAR* ddmacd(const int& l) { return &_ddmacd[l * _ng]; };
	__host__ __device__ inline XS_VAR* ddmaca(const int& l) { return &_ddmaca[l * _ng]; };
	__host__ __device__ inline XS_VAR* ddmacf(const int& l) { return &_ddmacf[l * _ng]; };
	__host__ __device__ inline XS_VAR* ddmack(const int& l) { return &_ddmack[l * _ng]; };
	__host__ __device__ inline XS_VAR* ddmacn(const int& l) { return &_ddmacn[l * _ng]; };
	__host__ __device__ inline XS_VAR* ddmacs(const int& l) { return &_ddmacs[l * _ng * _ng]; };
	__host__ __device__ inline XS_VAR* dmmacd(const int& ip, const int& l) { return &_dmmacd[l * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline XS_VAR* dmmaca(const int& ip, const int& l) { return &_dmmaca[l * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline XS_VAR* dmmacf(const int& ip, const int& l) { return &_dmmacf[l * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline XS_VAR* dmmack(const int& ip, const int& l) { return &_dmmack[l * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline XS_VAR* dmmacn(const int& ip, const int& l) { return &_dmmacn[l * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline XS_VAR* dmmacs(const int& ip, const int& l) { return &_dmmacs[l * _ng * _ng * NPTM + ip * _ng * _ng]; };
	__host__ __device__ inline XS_VAR* dfmacd(const int& l) { return &_dfmacd[l * _ng]; };
	__host__ __device__ inline XS_VAR* dfmaca(const int& l) { return &_dfmaca[l * _ng]; };
	__host__ __device__ inline XS_VAR* dfmacf(const int& l) { return &_dfmacf[l * _ng]; };
	__host__ __device__ inline XS_VAR* dfmack(const int& l) { return &_dfmack[l * _ng]; };
	__host__ __device__ inline XS_VAR* dfmacn(const int& l) { return &_dfmacn[l * _ng]; };
	__host__ __device__ inline XS_VAR* dfmacs(const int& l) { return &_dfmacs[l * _ng * _ng]; };

	__host__ __device__ inline XS_VAR* xsmacd0(const int& l) { return &_xsmacd0[l * _ng]; };
	__host__ __device__ inline XS_VAR* xsmaca0(const int& l) { return &_xsmaca0[l * _ng]; };
	__host__ __device__ inline XS_VAR* xsmacf0(const int& l) { return &_xsmacf0[l * _ng]; };
	__host__ __device__ inline XS_VAR* xsmack0(const int& l) { return &_xsmack0[l * _ng]; };
	__host__ __device__ inline XS_VAR* xsmacn0(const int& l) { return &_xsmacn0[l * _ng]; };
	__host__ __device__ inline XS_VAR* xsmacs0(const int& l) { return &_xsmacs0[l * _ng * _ng]; };

	__host__ __device__ inline XS_VAR* xdfmicd(const int& l) { return &_xdfmicd[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xddmicd(const int& l) { return &_xddmicd[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xdpmicd(const int& l) { return &_xdpmicd[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xdpmicf(const int& l) { return &_xdpmicf[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xdfmicf(const int& l) { return &_xdfmicf[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xddmicf(const int& l) { return &_xddmicf[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xdfmica(const int& l) { return &_xdfmica[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xddmica(const int& l) { return &_xddmica[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xdpmica(const int& l) { return &_xdpmica[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xdpmicn(const int& l) { return &_xdpmicn[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xdfmicn(const int& l) { return &_xdfmicn[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xddmicn(const int& l) { return &_xddmicn[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xdpmick(const int& l) { return &_xdpmick[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xdfmick(const int& l) { return &_xdfmick[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xddmick(const int& l) { return &_xddmick[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xdfmics(const int& l) { return &_xdfmics[l * _ng * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xddmics(const int& l) { return &_xddmics[l * _ng * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xdpmics(const int& l) { return &_xdpmics[l * _ng * _ng * NISO]; };

	__host__ __device__ inline XS_VAR* xdmmicd(const int& l) { return &_xdmmicd[l * _ng * NPTM * NISO]; };
	__host__ __device__ inline XS_VAR* xdmmicf(const int& l) { return &_xdmmicf[l * _ng * NPTM * NISO]; };
	__host__ __device__ inline XS_VAR* xdmmica(const int& l) { return &_xdmmica[l * _ng * NPTM * NISO]; };
	__host__ __device__ inline XS_VAR* xdmmicn(const int& l) { return &_xdmmicn[l * _ng * NPTM * NISO]; };
	__host__ __device__ inline XS_VAR* xdmmick(const int& l) { return &_xdmmick[l * _ng * NPTM * NISO]; };
	__host__ __device__ inline XS_VAR* xdmmics(const int& l) { return &_xdmmics[l * _ng * _ng * NPTM * NISO]; };

	__host__ __device__ inline XS_VAR* xsmicd(const int& l) { return &_xsmicd[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xsmica(const int& l) { return &_xsmica[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xsmics(const int& l) { return &_xsmics[l * _ng * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xsmicf(const int& l) { return &_xsmicf[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xsmick(const int& l) { return &_xsmick[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xsmicn(const int& l) { return &_xsmicn[l * _ng * NISO]; };

	__host__ __device__ inline XS_VAR* xsmicd0(const int& l) { return &_xsmicd0[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xsmica0(const int& l) { return &_xsmica0[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xsmicf0(const int& l) { return &_xsmicf0[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xsmick0(const int& l) { return &_xsmick0[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xsmicn0(const int& l) { return &_xsmicn0[l * _ng * NISO]; };
	__host__ __device__ inline XS_VAR* xsmics0(const int& l) { return &_xsmics0[l * _ng * _ng * NISO]; };

	__host__ __device__ inline XS_VAR* xehfp(const int& l) { return &_xehfp[l]; };


};
