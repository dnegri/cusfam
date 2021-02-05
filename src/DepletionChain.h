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


class DepletionChain : public Managed {
private:
    Geometry& _g;

    int _mnucl;
    int _nfcnt;

    int _nhvychn;
    int* _nheavy;                //(:)
    int* _ihvys;                //(:)
    int _nhvyids;
    int* _hvyids;                //(:,:)
    int* _reactype;              //(:,:)
    int* _hvyupd;                //(:,:)

    // 2-1. Fission Product by Actinide Isotope
    int _nfiso;
    int _nfpiso;
    int* _fiso;
    int* _fpiso;           //(:)
    float* _fyld;
    int _fpPM147;
    int _fpPM149;
    int _fpSM149;
    int _fpI135;
    int _fpXE145;
    float FRAC48 = 0.5277;


    // 2-2. Fission Product Chain Define
    int _nsm;
    int* _smids;         //(:)
    int _nxe;
    int* _xeids;          //(:)

    // 3. Decay Constant Define
    int _ndcy;
    int* _dcyids;         //(:)
    float* _dcnst;              //(:)

    float* _cap;
    float* _rem;
    float* _fis;
    float* _dcy;
    float* _tn2n;

    int ixe;
    int ism;

    float* _dnst;
    float* _burn;
    float* _h2on;


    float _b10ap;
    float _b10wp;
    float _b10fac;

public:
    __host__  __device__ DepletionChain();
    __host__  __device__ DepletionChain(Geometry& g);

    __host__ __device__ virtual ~DepletionChain();
    
    __host__ __device__ float& h2on(const int& l) { return _h2on[l]; };

    __host__ __device__ float& burn(const int& l) { return _burn[l]; };
    __host__ __device__ float& cap(const int& iiso, const int& l) { return _cap[l*_mnucl + iiso]; } ;
    __host__ __device__ float& rem(const int& iiso, const int& l) { return _rem[l*_mnucl + iiso]; } ;
    __host__ __device__ float& fis(const int& iiso, const int& l) { return _fis[l*_mnucl + iiso]; } ;
    __host__ __device__ float& dcy(const int& iiso, const int& l) { return _dcy[l*_mnucl + iiso]; } ;
    __host__ __device__ float& dnst(const int& iiso, const int& l) { return _dnst[l * NISO + iiso]; };
    __host__ __device__ float* dnst() { return _dnst; };
    __host__ __device__ float* burn() { return _burn; };

    __host__ __device__ int& nheavy(const int& ichn) { return _nheavy[ichn]; };
    __host__ __device__ int& ihchn(const int& step, const int& ichn) { return _hvyids[_ihvys[ichn]+ step]; };
    __host__ __device__ int& idpct(const int& step, const int& ichn) { return _hvyupd[_ihvys[ichn] + step]; };
    __host__ __device__ int& iptyp(const int& step, const int& ichn) { return _reactype[_ihvys[ichn] + step]; };
    __host__ __device__ float& fyld(const int& fpiso, const int& fiso) { return _fyld[fiso*_nfpiso + fpiso]; };


    __host__ __device__ void dep(const int& l, const float& tsec, const float* ati, float* atd, float* atavg);
    __host__ __device__ void deph(const int& l, const float& tsec, const float* ati, float* atd, float* atavg);
    __host__ __device__ void depsm(const int& l, const float& tsec, const float* ati, float* atd, float* atavg);
    __host__ __device__ void depxe(const int& l, const float& tsec, const float* ati, float* atd, float* atavg);
    __host__ __device__ void depp(const int& l, const float& tsec, const float* ati, float* atd, float* atavg);
    __host__ __device__ void pickData(const int& l, const float* xsmica, const float* xsmicf, const float* xsmic2n, const double* phi);

    __host__ __device__ void updateH2ODensity(const int& l, const float* dm, const float& ppm);


};