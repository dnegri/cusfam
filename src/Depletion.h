#pragma once

#include "pch.h"
#include "Geometry.h"

enum ChainType {
    IDX_DEFAULT,
    IDX_C,
    IDX_H,
    IDX_0,
};

enum ChainAction {
    LEAVE,
    UPDATE,
    HOLD
};

enum ChainReaction {
    R_CAP,
    R_DEC,
    R_N2N
};

enum XEType {
    XE_NO,
    XE_EQ,
    XE_TR
};

enum SMType {
    SM_NO,
    SM_TR
};


class Depletion : public Managed {
protected:
    Geometry& _g;

    int _nhvychn;
    int* _nheavy;                //(:)
    int* _ihvys;                //(:)
    int _nhvyids;
    int* _hvyids;                //(:,:)
    int* _reactype;              //(:,:)
    int* _hvyupd;                //(:,:)

    float FRAC48 = 0.5277;


    // 3. Decay Constant Define
    float* _dcy;

    float* _cap;
    float* _rem;
    float* _fis;
    float* _tn2n;

    float* _dnst;
    float* _dnst_new;
    float* _dnst_avg;
	float* _h2on;

	float  _cburn;
    float* _burn;
	float* _buconf;

    float _b10ap;
    float _b10wp;
    float _b10fac;
	float _totmass;

public:
    __host__ Depletion(Geometry& g);

    __host__ virtual ~Depletion();

    __host__  void init();

    
    __host__  __device__ Geometry& g() { return _g; };

    __host__ __device__ float& h2on(const int& l) { return _h2on[l]; };

    __host__ __device__ float& burn(const int& l) { return _burn[l]; };
    __host__ __device__ float& buconf(const int& l) { return _buconf[l]; };
    __host__ __device__ float& cap(const int& iiso, const int& l) { return _cap[l*NDEP + iiso]; } ;
    __host__ __device__ float& rem(const int& iiso, const int& l) { return _rem[l*NDEP + iiso]; } ;
    __host__ __device__ float& fis(const int& iiso, const int& l) { return _fis[l*NDEP + iiso]; } ;

    __host__ __device__ float& dcy(const int& iiso) { return _dcy[iiso]; };


	__host__ __device__ void multiplyDensity(const int& iiso, const float& factor);
	__host__ __device__ float& dnst(const int& iiso, const int& l) { return _dnst[l * NISO + iiso]; };
    __host__ __device__ float* dnst() { return _dnst; };
    __host__ __device__ float* dnst_new() { return _dnst_new; };
    __host__ __device__ float* dnst_avg() { return _dnst_avg; };

    __host__ __device__ float* burn() { return _burn; };
    __host__ __device__ float* h2on() { return _h2on; };
	__host__ __device__ float* buconf() { return _buconf; };
	__host__ __device__ float& totmass() { return _totmass; };

    __host__ __device__ int& nhchn(const int& ichn) { return _nheavy[ichn]; };
    __host__ __device__ int& ihchn(const int& step, const int& ichn) { return _hvyids[_ihvys[ichn]+ step]; };
    __host__ __device__ int& idpct(const int& step, const int& ichn) { return _hvyupd[_ihvys[ichn] + step]; };
    __host__ __device__ int& iptyp(const int& step, const int& ichn) { return _reactype[_ihvys[ichn] + step]; };
    __host__ __device__ const float& fyld(const ISO_FP& fpiso, const int& fiso) { return FPYLD[fiso*NFP + fpiso]; };

    __host__ void eqxe(const XS_VAR* xsmica, const XS_VAR* xsmicf, const SOL_VAR* flux, const float& fnorm);
    __host__ __device__ void eqxe(const int& l, const XS_VAR* xsmica, const XS_VAR* xsmicf, const SOL_VAR* flux, const float& fnorm);

    __host__ void dep(const float& tsec, const XEType& xeopt, const SMType& smopt, const float* power);
    __host__ __device__ void dep(const int& l, const float& tsec, const XEType& xeopt, const SMType& smopt, const float& power, float* ati, float* atd, float* atavg);
    __host__ __device__ void deph(const int& l, const float& tsec, const float* ati, float* atd, float* atavg);
    __host__ __device__ void depsm(const int& l, const float& tsec, const float* ati, float* atd, float* atavg);
    __host__ __device__ void depxe(const int& l, const float& tsec, const float* ati, float* atd, float* atavg);
    __host__ __device__ void depp(const int& l, const float& tsec, const float* ati, float* atd, float* atavg);

    __host__ void pickData(const XS_VAR* xsmica, const XS_VAR* xsmicf, const XS_VAR* xsmic2n, const SOL_VAR* flux, const float& fnorm);
    __host__ __device__ void pickData(const int& l, const XS_VAR* xsmica, const XS_VAR* xsmicf, const XS_VAR* xsmic2n, const SOL_VAR* flux, const float& fnorm);

	void updateB10Abundance(const float& b10ap);
	__host__ void updateH2ODensity(const float* dm, const float& ppm);
    __host__ __device__ void updateH2ODensity(const int& l, const float* dm, const float& ppm);

	const float& b10wp() {return _b10wp;};


};