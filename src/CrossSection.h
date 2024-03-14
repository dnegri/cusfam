#pragma once
#include "pch.h"
#include "ControlRod.h"
#include "JIArray.h"

using namespace dnegri::jiarray;

class CrossSection : public Managed {
protected:
	int _ng;
	int _nxy;
    int _nz;
	int _nxyz;
	int _lec;

	double* _xsnf;
	double* _xsdf;
	double* _xssf;
	double* _xstf;
	double* _xskf;
	double* _chif;
	double* _xsadf;

	double* _xehfp;

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
	double* _xdpmicf;   // (:,:,:,:,:)
	double* _xdfmicf;   // (:,:,:,:,:)
	double* _xdmmicf;   // (:,:,:,:,:)
	double* _xddmicf;   // (:,:,:,:,:)
	double* _xdfmics;   // (:,:,:,:,:,:)
	double* _xdfmica;   // (:,:,:,:,:)
	double* _xdfmicd;   // (:,:,:,:,:)
	double* _xdmmics;   // (:,:,:,:,:,:)
	double* _xdmmica;   // (:,:,:,:,:)
	double* _xdmmicd;   // (:,:,:,:,:)
	double* _xddmics;   // (:,:,:,:,:,:)
	double* _xddmica;   // (:,:,:,:,:)
	double* _xddmicd;   // (:,:,:,:,:)
	double* _xdpmics;   // (:,:,:,:,:,:)
	double* _xdpmica;   // (:,:,:,:,:)
	double* _xdpmicd;   // (:,:,:,:,:)
	double* _xdpmicn;   // (:,:,:,:,:)
	double* _xdfmicn;   // (:,:,:,:,:)
	double* _xdmmicn;   // (:,:,:,:,:)
	double* _xddmicn;   // (:,:,:,:,:)
	double* _xdpmick;   // (:,:,:,:,:)
	double* _xdfmick;   // (:,:,:,:,:)
	double* _xdmmick;   // (:,:,:,:,:)
	double* _xddmick;   // (:,:,:,:,:)

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
	double* _xsmicd; // (:,:,:,:)
	double* _xsmica; // (:,:,:,:)
	double* _xsmics; // (:,:,:,:,:)
	double* _xsmicf; // (:,:,:,:)
	double* _xsmick; // (:,:,:,:)
	double* _xsmicn; // (:,:,:,:)
	double* _xsmic2n; // (:,:)

	double* _xsmicd0; // (:,:,:,:)
	double* _xsmica0; // (:,:,:,:)
	double* _xsmics0; // (:,:,:,:,:)
	double* _xsmicf0; // (:,:,:,:)
	double* _xsmick0; // (:,:,:,:)
	double* _xsmicn0; // (:,:,:,:)

	double* _xsmacd0; // (:,:,:,:)
	double* _xsmaca0; // (:,:,:,:)
	double* _xsmacs0; // (:,:,:,:,:)
	double* _xsmacf0; // (:,:,:,:)
	double* _xsmack0; // (:,:,:,:)
	double* _xsmacn0; // (:,:,:,:)

	double* _dpmacd;
	double* _dpmaca;
	double* _dpmacf;
	double* _dpmack;
	double* _dpmacn;
	double* _dpmacs;
	double* _ddmacd;
	double* _ddmaca;
	double* _ddmacf;
	double* _ddmack;
	double* _ddmacn;
	double* _ddmacs;
	double* _dmmacd;
	double* _dmmaca;
	double* _dmmacf;
	double* _dmmack;
	double* _dmmacn;
	double* _dmmacs;
	double* _dfmacd;
	double* _dfmaca;
	double* _dfmacf;
	double* _dfmack;
	double* _dfmacn;
	double* _dfmacs;

	double* _delmac;

public:
	__host__ __device__ CrossSection() {
	};

	__host__ __device__ CrossSection(const int& ng, const int& nxy, const int& nz, const int& lec) {
		init(ng, nxy, nz,lec);
	}

	__host__ __device__ void init(const int& ng, const int& nxy, const int& nz, const int& lec) {
		_ng = ng;
		_nxyz = nxy*nz;
		_nxy = nxy;
        _nz = nz;
		_lec = lec;

		_xsnf = new double[_ng * _nxyz]{};
		_xsdf = new double[_ng * _nxyz]{};
		_xstf = new double[_ng * _nxyz]{};
		_xskf = new double[_ng * _nxyz]{};
		_chif = new double[_ng * _nxyz]{};
		_xssf = new double[_ng * _ng * _nxyz]{};
		_xsadf = new double[_ng * _nxyz]{};
		_xehfp = new double[ _nxyz]{};

		_xsmacd0 = new double[_ng * _nxyz]; // (:,:,:,:)
		_xsmaca0 = new double[_ng * _nxyz]; // (:,:,:,:)
		_xsmacs0 = new double[_ng * _ng * _nxyz]; // (:,:,:,:,:)
		_xsmacf0 = new double[_ng * _nxyz]; // (:,:,:,:)
		_xsmack0 = new double[_ng * _nxyz]; // (:,:,:,:)
		_xsmacn0 = new double[_ng * _nxyz]; // (:,:,:,:)


		_xsmicd = new double[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmica = new double[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmics = new double[_ng * _ng * NISO * _nxyz]{}; // (:,:,:,:,:)
		_xsmicf = new double[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmick = new double[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmicn = new double[_ng * NISO * _nxyz]{}; // (:,:,:,:)

		_xsmic2n = new double[_ng * NISO * _nxyz]{}; // (:,:)
		_xsmicd0 = new double[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmica0 = new double[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmics0 = new double[_ng * _ng * NISO * _nxyz]{}; // (:,:,:,:,:)
		_xsmicf0 = new double[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmick0 = new double[_ng * NISO * _nxyz]{}; // (:,:,:,:)
		_xsmicn0 = new double[_ng * NISO * _nxyz]{}; // (:,:,:,:)

		_xdfmicd = new double[_ng * NISO * _nxyz]{};
		_xdfmica = new double[_ng * NISO * _nxyz]{};
		_xddmicd = new double[_ng * NISO * _nxyz]{};
		_xddmica = new double[_ng * NISO * _nxyz]{};
		_xdpmicd = new double[_ng * NISO * _nxyz]{};
		_xdpmica = new double[_ng * NISO * _nxyz]{};
		_xdmmicd = new double[_ng * NPTM * NISO * _nxyz]{};
		_xdmmica = new double[_ng * NPTM * NISO * _nxyz]{};
		_xddmics = new double[_ng * _ng * NISO * _nxyz]{};
		_xdpmics = new double[_ng * _ng * NISO * _nxyz]{};
		_xdfmics = new double[_ng * _ng * NISO * _nxyz]{};
		_xdmmics = new double[_ng * _ng * NPTM * NISO * _nxyz]{};

		_xdfmicn = new double[_ng * NISO * _nxyz]{};
		_xddmicn = new double[_ng * NISO * _nxyz]{};
		_xdpmicn = new double[_ng * NISO * _nxyz]{};
		_xdmmicn = new double[_ng * NPTM * NISO * _nxyz]{};
		_xdpmicf = new double[_ng * NISO * _nxyz]{};
		_xdfmicf = new double[_ng * NISO * _nxyz]{};
		_xddmicf = new double[_ng * NISO * _nxyz]{};
		_xdmmicf = new double[_ng * NPTM * NISO * _nxyz]{};
		_xdpmick = new double[_ng * NISO * _nxyz]{};
		_xdfmick = new double[_ng * NISO * _nxyz]{};
		_xddmick = new double[_ng * NISO * _nxyz]{};
		_xdmmick = new double[_ng * NPTM * NISO * _nxyz]{};


		_dpmacd = new double[_nxyz * _ng]{};
		_dpmaca = new double[_nxyz * _ng]{};
		_dpmacf = new double[_nxyz * _ng]{};
		_dpmack = new double[_nxyz * _ng]{};
		_dpmacn = new double[_nxyz * _ng]{};
		_dpmacs = new double[_nxyz * _ng * _ng]{};
		_ddmacd = new double[_nxyz * _ng]{};
		_ddmaca = new double[_nxyz * _ng]{};
		_ddmacf = new double[_nxyz * _ng]{};
		_ddmack = new double[_nxyz * _ng]{};
		_ddmacn = new double[_nxyz * _ng]{};
		_ddmacs = new double[_nxyz * _ng * _ng]{};
		_dmmacd = new double[_nxyz * NPTM * _ng]{};
		_dmmaca = new double[_nxyz * NPTM * _ng]{};
		_dmmacf = new double[_nxyz * NPTM * _ng]{};
		_dmmack = new double[_nxyz * NPTM * _ng]{};
		_dmmacn = new double[_nxyz * NPTM * _ng]{};
		_dmmacs = new double[_nxyz * NPTM * _ng * _ng]{};
		_dfmacd = new double[_nxyz * _ng]{};
		_dfmaca = new double[_nxyz * _ng]{};
		_dfmacf = new double[_nxyz * _ng]{};
		_dfmack = new double[_nxyz * _ng]{};
		_dfmacn = new double[_nxyz * _ng]{};
		_dfmacs = new double[_nxyz * _ng * _ng]{};

		_delmac = new double[_ng * _nxyz]{};


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

		delete[] _delmac;
	}


	__host__ __device__ const int& ng() const { return _ng; }
	__host__ __device__ const int& nxyz() const { return _nxyz; }

	__host__ __device__ void updateMacroXS(const int& l, float* dnst);
	__host__ void updateMacroXS(float* dnst);

	__host__ __device__ void updateXS(const int& l, const float* dnst, const float& dppm, const float& dtf, const float& dtm);
    __host__ void updateXS(const float* dnst, const float* dppm, const float* dtf, const float* dtm);

	__host__ __device__ void updateDBC(const int& l,  const float* dnst, const float& tf, const float& dppm, const float& dtf, const float& dtm);
	__host__ void updateDBC(const float* dnst, const float* tf, const float* dppm, const float* dtf, const float* dtm);


	__host__ __device__ void updateXenonXS(const int& l, const float& dppm, const float& dtf, const float& dtm);
	__host__ void updateXenonXS(const float* dppm, const float* dtf, const float* dtm);

	__host__ __device__ void updateXeSmXS(const int& l, const float& dppm, const float& dtf, const float& dtm);
	__host__ void updateXeSmXS(const float* dppm, const float* dtf, const float* dtm);

	__host__ __device__ void updateDepletionXS(const int& l, const float& dppm, const float& dtf, const float& dtm);
	__host__ void updateDepletionXS(const float* dppm, const float* dtf, const float* dtm);

	__host__ __device__ void updateRodXS(const int& l, const int& iso_rod, const float& ratio, const float& dppm, const float& dtf, const float& dtm);
	__host__ __device__	void updateRodXSInRefl(const int& lFuel, const int& lRefl, const int& iso_rod, const float& ratio, const float& dppm, const float& dtf, const float& dtm);
    __host__ void updateRodXS(ControlRod& r, const float* dppm, const float* dtf, const float* dtm);

    void updateShapeMatchXS(const double2& dxsa);

    __host__ __device__ inline double& xsnf(const int& ig, const int& l) { return _xsnf[l * _ng + ig]; };
	__host__ __device__ inline double& xsdf(const int& ig, const int& l) { return _xsdf[l * _ng + ig]; };
	__host__ __device__ inline double& xssf(const int& igs, const int& ige, const int& l) { return _xssf[l * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline double& xstf(const int& ig, const int& l) { return _xstf[l * _ng + ig]; };
	__host__ __device__ inline double& xskf(const int& ig, const int& l) { return _xskf[l * _ng + ig]; };
	__host__ __device__ inline double& chif(const int& ig, const int& l) { return _chif[l * _ng + ig]; };
	__host__ __device__ inline double& xsadf(const int& ig, const int& l) { return _xsadf[l * _ng + ig]; };

	__host__ __device__ inline double& xdfmicd(const int& ig, const int& iiso, const int& l) { return _xdfmicd[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xddmicd(const int& ig, const int& iiso, const int& l) { return _xddmicd[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xdpmicd(const int& ig, const int& iiso, const int& l) { return _xdpmicd[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xdpmicf(const int& ig, const int& iiso, const int& l) { return _xdpmicf[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xdfmicf(const int& ig, const int& iiso, const int& l) { return _xdfmicf[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xddmicf(const int& ig, const int& iiso, const int& l) { return _xddmicf[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xdfmica(const int& ig, const int& iiso, const int& l) { return _xdfmica[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xddmica(const int& ig, const int& iiso, const int& l) { return _xddmica[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xdpmica(const int& ig, const int& iiso, const int& l) { return _xdpmica[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xdpmicn(const int& ig, const int& iiso, const int& l) { return _xdpmicn[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xdfmicn(const int& ig, const int& iiso, const int& l) { return _xdfmicn[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xddmicn(const int& ig, const int& iiso, const int& l) { return _xddmicn[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xdpmick(const int& ig, const int& iiso, const int& l) { return _xdpmick[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xdfmick(const int& ig, const int& iiso, const int& l) { return _xdfmick[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xddmick(const int& ig, const int& iiso, const int& l) { return _xddmick[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xdfmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xdfmics[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline double& xddmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xddmics[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline double& xdpmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xdpmics[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };

	__host__ __device__ inline double& xdmmicd(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmicd[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline double& xdmmicf(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmicf[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline double& xdmmica(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmica[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline double& xdmmicn(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmicn[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline double& xdmmick(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmick[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline double& xdmmics(const int& igs, const int& ige, const int& ip, const int& iiso, const int& l) { return _xdmmics[l * _ng * _ng * NPTM * NISO + iiso * _ng * _ng * NPTM + ip * _ng * _ng + ige * _ng + igs]; };

	__host__ __device__ inline double& xsmic2n(const int& ig, const int& l) { return _xsmic2n[l * _ng + ig]; };
	__host__ __device__ inline double& xsmicd(const int& ig, const int& iiso, const int& l) { return _xsmicd[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xsmica(const int& ig, const int& iiso, const int& l) { return _xsmica[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xsmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xsmics[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline double& xsmicf(const int& ig, const int& iiso, const int& l) { return _xsmicf[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xsmick(const int& ig, const int& iiso, const int& l) { return _xsmick[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xsmicn(const int& ig, const int& iiso, const int& l) { return _xsmicn[l * _ng * NISO + iiso * _ng + ig]; };

	__host__ __device__ inline double& xsmicd0(const int& ig, const int& iiso, const int& l) { return _xsmicd0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xsmica0(const int& ig, const int& iiso, const int& l) { return _xsmica0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xsmicf0(const int& ig, const int& iiso, const int& l) { return _xsmicf0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xsmick0(const int& ig, const int& iiso, const int& l) { return _xsmick0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xsmicn0(const int& ig, const int& iiso, const int& l) { return _xsmicn0[l * _ng * NISO + iiso * _ng + ig]; };
	__host__ __device__ inline double& xsmics0(const int& igs, const int& ige, const int& iiso, const int& l) { return _xsmics0[l * _ng * _ng * NISO + iiso * _ng * _ng + ige * _ng + igs]; };


	__host__ __device__ inline double& dpmacd(const int& ig, const int& l) { return _dpmacd[l * _ng + ig]; };
	__host__ __device__ inline double& dpmaca(const int& ig, const int& l) { return _dpmaca[l * _ng + ig]; };
	__host__ __device__ inline double& dpmacf(const int& ig, const int& l) { return _dpmacf[l * _ng + ig]; };
	__host__ __device__ inline double& dpmack(const int& ig, const int& l) { return _dpmack[l * _ng + ig]; };
	__host__ __device__ inline double& dpmacn(const int& ig, const int& l) { return _dpmacn[l * _ng + ig]; };
	__host__ __device__ inline double& dpmacs(const int& igs, const int& ige, const int& l) { return _dpmacs[l * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline double& ddmacd(const int& ig, const int& l) { return _ddmacd[l * _ng + ig]; };
	__host__ __device__ inline double& ddmaca(const int& ig, const int& l) { return _ddmaca[l * _ng + ig]; };
	__host__ __device__ inline double& ddmacf(const int& ig, const int& l) { return _ddmacf[l * _ng + ig]; };
	__host__ __device__ inline double& ddmack(const int& ig, const int& l) { return _ddmack[l * _ng + ig]; };
	__host__ __device__ inline double& ddmacn(const int& ig, const int& l) { return _ddmacn[l * _ng + ig]; };
	__host__ __device__ inline double& ddmacs(const int& igs, const int& ige, const int& l) { return _ddmacs[l * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline double& dmmacd(const int& ig, const int& ip, const int& l) { return _dmmacd[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline double& dmmaca(const int& ig, const int& ip, const int& l) { return _dmmaca[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline double& dmmacf(const int& ig, const int& ip, const int& l) { return _dmmacf[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline double& dmmack(const int& ig, const int& ip, const int& l) { return _dmmack[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline double& dmmacn(const int& ig, const int& ip, const int& l) { return _dmmacn[l * _ng * NPTM + ip * _ng + ig]; };
	__host__ __device__ inline double& dmmacs(const int& igs, const int& ige, const int& ip, const int& l) { return _dmmacs[l * _ng * _ng * NPTM + ip * _ng * _ng + ige * _ng + igs]; };
	__host__ __device__ inline double& dfmacd(const int& ig, const int& l) { return _dfmacd[l * _ng + ig]; };
	__host__ __device__ inline double& dfmaca(const int& ig, const int& l) { return _dfmaca[l * _ng + ig]; };
	__host__ __device__ inline double& dfmacf(const int& ig, const int& l) { return _dfmacf[l * _ng + ig]; };
	__host__ __device__ inline double& dfmack(const int& ig, const int& l) { return _dfmack[l * _ng + ig]; };
	__host__ __device__ inline double& dfmacn(const int& ig, const int& l) { return _dfmacn[l * _ng + ig]; };
	__host__ __device__ inline double& dfmacs(const int& igs, const int& ige, const int& l) { return _dfmacs[l * _ng * _ng + ige * _ng + igs]; };

	__host__ __device__ inline double& xsmacd0(const int& ig, const int& l) { return _xsmacd0[l * _ng + ig]; };
	__host__ __device__ inline double& xsmaca0(const int& ig, const int& l) { return _xsmaca0[l * _ng + ig]; };
	__host__ __device__ inline double& xsmacf0(const int& ig, const int& l) { return _xsmacf0[l * _ng + ig]; };
	__host__ __device__ inline double& xsmack0(const int& ig, const int& l) { return _xsmack0[l * _ng + ig]; };
	__host__ __device__ inline double& xsmacn0(const int& ig, const int& l) { return _xsmacn0[l * _ng + ig]; };
	__host__ __device__ inline double& xsmacs0(const int& igs, const int& ige, const int& l) { return _xsmacs0[l * _ng * _ng + ige * _ng + igs]; };

	__host__ __device__ inline double& delmac(const int& ig, const int& l) { return _delmac[l * _ng + ig]; };
	__host__ __device__ inline double* delmac() { return _delmac; };

	const double* xsnf() const { return _xsnf; };
	const double* xsdf() const { return _xsdf; };
	const double* xstf() const { return _xstf; };
	const double* xskf() const { return _xskf; };
	const double* chif() const { return _chif; };
	const double* xssf() const { return _xssf; };
	const double* xsadf() const { return _xsadf; };
	const double* xsmacd0() const { return _xsmacd0; };
	const double* xsmaca0() const { return _xsmaca0; };
	const double* xsmacs0() const { return _xsmacs0; };
	const double* xsmacf0() const { return _xsmacf0; };
	const double* xsmack0() const { return _xsmack0; };
	const double* xsmacn0() const { return _xsmacn0; };
	const double* xsmicd() const { return _xsmicd; };
	const double* xsmica() const { return _xsmica; };
	const double* xsmics() const { return _xsmics; };
	const double* xsmicf() const { return _xsmicf; };
	const double* xsmick() const { return _xsmick; };
	const double* xsmicn() const { return _xsmicn; };
	const double* xsmic2n() const { return _xsmic2n; };
	const double* xsmicd0() const { return _xsmicd0; };
	const double* xsmica0() const { return _xsmica0; };
	const double* xsmics0() const { return _xsmics0; };
	const double* xsmicf0() const { return _xsmicf0; };
	const double* xsmick0() const { return _xsmick0; };
	const double* xsmicn0() const { return _xsmicn0; };
	const double* xdfmicd() const { return _xdfmicd; };
	const double* xdfmica() const { return _xdfmica; };
	const double* xddmicd() const { return _xddmicd; };
	const double* xddmica() const { return _xddmica; };
	const double* xdpmicd() const { return _xdpmicd; };
	const double* xdpmica() const { return _xdpmica; };
	const double* xdmmicd() const { return _xdmmicd; };
	const double* xdmmica() const { return _xdmmica; };
	const double* xddmics() const { return _xddmics; };
	const double* xdpmics() const { return _xdpmics; };
	const double* xdfmics() const { return _xdfmics; };
	const double* xdmmics() const { return _xdmmics; };
	const double* xdfmicn() const { return _xdfmicn; };
	const double* xddmicn() const { return _xddmicn; };
	const double* xdpmicn() const { return _xdpmicn; };
	const double* xdmmicn() const { return _xdmmicn; };
	const double* xdpmicf() const { return _xdpmicf; };
	const double* xdfmicf() const { return _xdfmicf; };
	const double* xddmicf() const { return _xddmicf; };
	const double* xdmmicf() const { return _xdmmicf; };
	const double* xdpmick() const { return _xdpmick; };
	const double* xdfmick() const { return _xdfmick; };
	const double* xddmick() const { return _xddmick; };
	const double* xdmmick() const { return _xdmmick; };
	const double* dpmacd() const { return _dpmacd; };
	const double* dpmaca() const { return _dpmaca; };
	const double* dpmacf() const { return _dpmacf; };
	const double* dpmack() const { return _dpmack; };
	const double* dpmacn() const { return _dpmacn; };
	const double* dpmacs() const { return _dpmacs; };
	const double* ddmacd() const { return _ddmacd; };
	const double* ddmaca() const { return _ddmaca; };
	const double* ddmacf() const { return _ddmacf; };
	const double* ddmack() const { return _ddmack; };
	const double* ddmacn() const { return _ddmacn; };
	const double* ddmacs() const { return _ddmacs; };
	const double* dmmacd() const { return _dmmacd; };
	const double* dmmaca() const { return _dmmaca; };
	const double* dmmacf() const { return _dmmacf; };
	const double* dmmack() const { return _dmmack; };
	const double* dmmacn() const { return _dmmacn; };
	const double* dmmacs() const { return _dmmacs; };
	const double* dfmacd() const { return _dfmacd; };
	const double* dfmaca() const { return _dfmaca; };
	const double* dfmacf() const { return _dfmacf; };
	const double* dfmack() const { return _dfmack; };
	const double* dfmacn() const { return _dfmacn; };
	const double* dfmacs() const { return _dfmacs; };

	__host__ __device__ inline double* xsnf(const int& l) { return &_xsnf[l * _ng]; };
	__host__ __device__ inline double* xsdf(const int& l) { return &_xsdf[l * _ng]; };
	__host__ __device__ inline double* xssf(const int& l) { return &_xssf[l * _ng * _ng]; };
	__host__ __device__ inline double* xstf(const int& l) { return &_xstf[l * _ng]; };
	__host__ __device__ inline double* xskf(const int& l) { return &_xskf[l * _ng]; };
	__host__ __device__ inline double* chif(const int& l) { return &_chif[l * _ng]; };
	__host__ __device__ inline double* xsadf(const int& l) { return &_xsadf[l * _ng]; };

	__host__ __device__ inline double* xdfmicd(const int& iiso, const int& l) { return &_xdfmicd[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xddmicd(const int& iiso, const int& l) { return &_xddmicd[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xdpmicd(const int& iiso, const int& l) { return &_xdpmicd[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xdpmicf(const int& iiso, const int& l) { return &_xdpmicf[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xdfmicf(const int& iiso, const int& l) { return &_xdfmicf[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xddmicf(const int& iiso, const int& l) { return &_xddmicf[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xdfmica(const int& iiso, const int& l) { return &_xdfmica[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xddmica(const int& iiso, const int& l) { return &_xddmica[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xdpmica(const int& iiso, const int& l) { return &_xdpmica[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xdpmicn(const int& iiso, const int& l) { return &_xdpmicn[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xdfmicn(const int& iiso, const int& l) { return &_xdfmicn[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xddmicn(const int& iiso, const int& l) { return &_xddmicn[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xdpmick(const int& iiso, const int& l) { return &_xdpmick[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xdfmick(const int& iiso, const int& l) { return &_xdfmick[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xddmick(const int& iiso, const int& l) { return &_xddmick[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xdfmics(const int& iiso, const int& l) { return &_xdfmics[l * _ng * _ng * NISO + iiso * _ng * _ng]; };
	__host__ __device__ inline double* xddmics(const int& iiso, const int& l) { return &_xddmics[l * _ng * _ng * NISO + iiso * _ng * _ng]; };
	__host__ __device__ inline double* xdpmics(const int& iiso, const int& l) { return &_xdpmics[l * _ng * _ng * NISO + iiso * _ng * _ng]; };

	__host__ __device__ inline double* xdmmicd(const int& ip, const int& iiso, const int& l) { return &_xdmmicd[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline double* xdmmicf(const int& ip, const int& iiso, const int& l) { return &_xdmmicf[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline double* xdmmica(const int& ip, const int& iiso, const int& l) { return &_xdmmica[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline double* xdmmicn(const int& ip, const int& iiso, const int& l) { return &_xdmmicn[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline double* xdmmick(const int& ip, const int& iiso, const int& l) { return &_xdmmick[l * _ng * NPTM * NISO + iiso * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline double* xdmmics(const int& ip, const int& iiso, const int& l) { return &_xdmmics[l * _ng * _ng * NPTM * NISO + iiso * _ng * _ng * NPTM + ip * _ng * _ng]; };

	__host__ __device__ inline double* xsmic2n(const int& l) { return &_xsmic2n[l * _ng]; };
	__host__ __device__ inline double* xsmicd(const int& iiso, const int& l) { return &_xsmicd[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xsmica(const int& iiso, const int& l) { return &_xsmica[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xsmics(const int& iiso, const int& l) { return &_xsmics[l * _ng * _ng * NISO + iiso * _ng * _ng]; };
	__host__ __device__ inline double* xsmicf(const int& iiso, const int& l) { return &_xsmicf[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xsmick(const int& iiso, const int& l) { return &_xsmick[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xsmicn(const int& iiso, const int& l) { return &_xsmicn[l * _ng * NISO + iiso * _ng]; };

	__host__ __device__ inline double* xsmicd0(const int& iiso, const int& l) { return &_xsmicd0[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xsmica0(const int& iiso, const int& l) { return &_xsmica0[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xsmicf0(const int& iiso, const int& l) { return &_xsmicf0[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xsmick0(const int& iiso, const int& l) { return &_xsmick0[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xsmicn0(const int& iiso, const int& l) { return &_xsmicn0[l * _ng * NISO + iiso * _ng]; };
	__host__ __device__ inline double* xsmics0(const int& iiso, const int& l) { return &_xsmics0[l * _ng * _ng * NISO + iiso * _ng * _ng]; };


	__host__ __device__ inline double* dpmacd(const int& l) { return &_dpmacd[l * _ng]; };
	__host__ __device__ inline double* dpmaca(const int& l) { return &_dpmaca[l * _ng]; };
	__host__ __device__ inline double* dpmacf(const int& l) { return &_dpmacf[l * _ng]; };
	__host__ __device__ inline double* dpmack(const int& l) { return &_dpmack[l * _ng]; };
	__host__ __device__ inline double* dpmacn(const int& l) { return &_dpmacn[l * _ng]; };
	__host__ __device__ inline double* dpmacs(const int& l) { return &_dpmacs[l * _ng * _ng]; };
	__host__ __device__ inline double* ddmacd(const int& l) { return &_ddmacd[l * _ng]; };
	__host__ __device__ inline double* ddmaca(const int& l) { return &_ddmaca[l * _ng]; };
	__host__ __device__ inline double* ddmacf(const int& l) { return &_ddmacf[l * _ng]; };
	__host__ __device__ inline double* ddmack(const int& l) { return &_ddmack[l * _ng]; };
	__host__ __device__ inline double* ddmacn(const int& l) { return &_ddmacn[l * _ng]; };
	__host__ __device__ inline double* ddmacs(const int& l) { return &_ddmacs[l * _ng * _ng]; };
	__host__ __device__ inline double* dmmacd(const int& ip, const int& l) { return &_dmmacd[l * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline double* dmmaca(const int& ip, const int& l) { return &_dmmaca[l * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline double* dmmacf(const int& ip, const int& l) { return &_dmmacf[l * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline double* dmmack(const int& ip, const int& l) { return &_dmmack[l * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline double* dmmacn(const int& ip, const int& l) { return &_dmmacn[l * _ng * NPTM + ip * _ng]; };
	__host__ __device__ inline double* dmmacs(const int& ip, const int& l) { return &_dmmacs[l * _ng * _ng * NPTM + ip * _ng * _ng]; };
	__host__ __device__ inline double* dfmacd(const int& l) { return &_dfmacd[l * _ng]; };
	__host__ __device__ inline double* dfmaca(const int& l) { return &_dfmaca[l * _ng]; };
	__host__ __device__ inline double* dfmacf(const int& l) { return &_dfmacf[l * _ng]; };
	__host__ __device__ inline double* dfmack(const int& l) { return &_dfmack[l * _ng]; };
	__host__ __device__ inline double* dfmacn(const int& l) { return &_dfmacn[l * _ng]; };
	__host__ __device__ inline double* dfmacs(const int& l) { return &_dfmacs[l * _ng * _ng]; };

	__host__ __device__ inline double* xsmacd0(const int& l) { return &_xsmacd0[l * _ng]; };
	__host__ __device__ inline double* xsmaca0(const int& l) { return &_xsmaca0[l * _ng]; };
	__host__ __device__ inline double* xsmacf0(const int& l) { return &_xsmacf0[l * _ng]; };
	__host__ __device__ inline double* xsmack0(const int& l) { return &_xsmack0[l * _ng]; };
	__host__ __device__ inline double* xsmacn0(const int& l) { return &_xsmacn0[l * _ng]; };
	__host__ __device__ inline double* xsmacs0(const int& l) { return &_xsmacs0[l * _ng * _ng]; };

	__host__ __device__ inline double* xdfmicd(const int& l) { return &_xdfmicd[l * _ng * NISO]; };
	__host__ __device__ inline double* xddmicd(const int& l) { return &_xddmicd[l * _ng * NISO]; };
	__host__ __device__ inline double* xdpmicd(const int& l) { return &_xdpmicd[l * _ng * NISO]; };
	__host__ __device__ inline double* xdpmicf(const int& l) { return &_xdpmicf[l * _ng * NISO]; };
	__host__ __device__ inline double* xdfmicf(const int& l) { return &_xdfmicf[l * _ng * NISO]; };
	__host__ __device__ inline double* xddmicf(const int& l) { return &_xddmicf[l * _ng * NISO]; };
	__host__ __device__ inline double* xdfmica(const int& l) { return &_xdfmica[l * _ng * NISO]; };
	__host__ __device__ inline double* xddmica(const int& l) { return &_xddmica[l * _ng * NISO]; };
	__host__ __device__ inline double* xdpmica(const int& l) { return &_xdpmica[l * _ng * NISO]; };
	__host__ __device__ inline double* xdpmicn(const int& l) { return &_xdpmicn[l * _ng * NISO]; };
	__host__ __device__ inline double* xdfmicn(const int& l) { return &_xdfmicn[l * _ng * NISO]; };
	__host__ __device__ inline double* xddmicn(const int& l) { return &_xddmicn[l * _ng * NISO]; };
	__host__ __device__ inline double* xdpmick(const int& l) { return &_xdpmick[l * _ng * NISO]; };
	__host__ __device__ inline double* xdfmick(const int& l) { return &_xdfmick[l * _ng * NISO]; };
	__host__ __device__ inline double* xddmick(const int& l) { return &_xddmick[l * _ng * NISO]; };
	__host__ __device__ inline double* xdfmics(const int& l) { return &_xdfmics[l * _ng * _ng * NISO]; };
	__host__ __device__ inline double* xddmics(const int& l) { return &_xddmics[l * _ng * _ng * NISO]; };
	__host__ __device__ inline double* xdpmics(const int& l) { return &_xdpmics[l * _ng * _ng * NISO]; };

	__host__ __device__ inline double* xdmmicd(const int& l) { return &_xdmmicd[l * _ng * NPTM * NISO]; };
	__host__ __device__ inline double* xdmmicf(const int& l) { return &_xdmmicf[l * _ng * NPTM * NISO]; };
	__host__ __device__ inline double* xdmmica(const int& l) { return &_xdmmica[l * _ng * NPTM * NISO]; };
	__host__ __device__ inline double* xdmmicn(const int& l) { return &_xdmmicn[l * _ng * NPTM * NISO]; };
	__host__ __device__ inline double* xdmmick(const int& l) { return &_xdmmick[l * _ng * NPTM * NISO]; };
	__host__ __device__ inline double* xdmmics(const int& l) { return &_xdmmics[l * _ng * _ng * NPTM * NISO]; };

	__host__ __device__ inline double* xsmicd(const int& l) { return &_xsmicd[l * _ng * NISO]; };
	__host__ __device__ inline double* xsmica(const int& l) { return &_xsmica[l * _ng * NISO]; };
	__host__ __device__ inline double* xsmics(const int& l) { return &_xsmics[l * _ng * _ng * NISO]; };
	__host__ __device__ inline double* xsmicf(const int& l) { return &_xsmicf[l * _ng * NISO]; };
	__host__ __device__ inline double* xsmick(const int& l) { return &_xsmick[l * _ng * NISO]; };
	__host__ __device__ inline double* xsmicn(const int& l) { return &_xsmicn[l * _ng * NISO]; };

	__host__ __device__ inline double* xsmicd0(const int& l) { return &_xsmicd0[l * _ng * NISO]; };
	__host__ __device__ inline double* xsmica0(const int& l) { return &_xsmica0[l * _ng * NISO]; };
	__host__ __device__ inline double* xsmicf0(const int& l) { return &_xsmicf0[l * _ng * NISO]; };
	__host__ __device__ inline double* xsmick0(const int& l) { return &_xsmick0[l * _ng * NISO]; };
	__host__ __device__ inline double* xsmicn0(const int& l) { return &_xsmicn0[l * _ng * NISO]; };
	__host__ __device__ inline double* xsmics0(const int& l) { return &_xsmics0[l * _ng * _ng * NISO]; };

	__host__ __device__ inline double* xehfp(const int& l) { return &_xehfp[l]; };


};
