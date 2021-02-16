#pragma once

#include "pch.h"
#include "Geometry.h"
#include "SteamTable.h"

class Feedback : public Managed{

protected:
    Geometry& _g;
    SteamTable& _steam;

    bool _tffrz;
    bool _tmfrz;
    bool _dmfrz;

    float _heatfrac = 1.0;

    float _hin;
    float _din;
    float _tin;

    float* _chflow;

    float* _ppm0;
    float* _stf0;
    float* _tm0;
    float* _dm0;

    float* _tf;
    float* _tm;
    float* _dm;

    float* _dtf;
    float* _dtm;
    float* _ddm;
    float* _dppm;

    int* _fueltype;
    float* _frodn;
    int _nft;
    int* _ntfpow;
    int* _ntfbu;
    float* _tfpow;
    float* _tfbu;
    float* _tftable;

public:
    __host__ __device__ Feedback(Geometry& g, SteamTable& steam);
    __host__ __device__ virtual ~Feedback();

    __host__ __device__ Geometry& g() {return _g;};
    __host__ __device__ void allocate();
    __host__ __device__ void initTFTable(const int& nft);

    __host__ __device__ float* tf() { return _tf; };
    __host__ __device__ float* tm() { return _tm; };
    __host__ __device__ float* dm() { return _dm; };
    __host__ __device__ float* ppm0() { return _ppm0; };
    __host__ __device__ float* stf0() { return _stf0; };
    __host__ __device__ float* tm0() { return _tm0; };
    __host__ __device__ float* dm0() { return _dm0; };
    __host__ __device__ float* chflow() { return _chflow; };
    __host__ __device__ int* fueltype() { return _fueltype; };
    __host__ __device__ float* frodn() { return _frodn; };
    __host__ __device__ float* dppm() { return _dppm; };
    __host__ __device__ float* dtf() { return _dtf; };
    __host__ __device__ float* dtm() { return _dtm; };
    __host__ __device__ float* ddm() { return _ddm; };

    __host__ __device__ float& ppm0(const int& l) { return _ppm0[l]; };
    __host__ __device__ float& stf0(const int& l) { return _stf0[l]; };
    __host__ __device__ float& tm0(const int& l) { return _tm0[l]; };
    __host__ __device__ float& dm0(const int& l) { return _dm0[l]; };
    __host__ __device__ float& dppm(const int& l) { return _dppm[l]; };
    __host__ __device__ float& dtf(const int& l) { return _dtf[l]; };
    __host__ __device__ float& dtm(const int& l) { return _dtm[l]; };
    __host__ __device__ float& ddm(const int& l) { return _ddm[l]; };

    __host__ __device__ float& tf(const int& l) { return _tf[l]; };
    __host__ __device__ float& tm(const int& l) { return _tm[l]; };
    __host__ __device__ float& dm(const int& l) { return _dm[l]; };
    __host__ __device__ float& tm(const int& l2d, const int& k) { return _tm[k*_g.nxy()+l2d]; };
    __host__ __device__ float& dm(const int& l2d, const int& k) { return _dm[k*_g.nxy()+l2d]; };


    __host__ __device__ float& chflow(const int& l2d) { return _chflow[l2d]; };
    __host__ __device__ void updateTf(const int& l, const float* pow, const float* bu, float heatfrac);
    __host__ __device__ void updateTm(const int& l2d, const float* pow, float hin, float tin, float din, int& nboiling);
    __host__ __device__ void updateTin(const float& tin);
    __host__ __device__ void updatePressure(const float& press);

    __host__ void updateTf(const float* power, const float* burnup);
    __host__ void updateTm(const float* power, int& nboiling);
    __host__ void updatePPM(const float& ppm);
    __host__  void initDelta(const float& ppm);

    __host__ __device__ int& nft() { return _nft; };
    __host__ __device__ int& fueltype(const int& l) { return _fueltype[l]; };
    __host__ __device__ float& frodn(const int& l2d) { return _frodn[l2d]; };

    __host__ __device__ int& ntfbu(const int& ift) { return _ntfbu[ift]; };
    __host__ __device__ int& ntfpow(const int& ift) { return _ntfpow[ift]; };

    __host__ __device__ float& tfbu(const int& ip, const int& ift) { return _tfbu[ift*TF_POINT+ip]; };
    __host__ __device__ float& tfpow(const int& ip, const int& ift) { return _tfpow[ift * TF_POINT + ip]; };
    __host__ __device__ float& tftable(const int& ibu, const int& ipow, const int& ift) { return _tftable[ift * TF_POINT * TF_POINT + ipow* TF_POINT + ibu]; };


    __host__ __device__ bool& tffrz() { return _tffrz; };
    __host__ __device__ bool& tmfrz() { return _tmfrz; };
    __host__ __device__ bool& dmfrz() { return _dmfrz; };
    __host__ __device__ float& heatfrac() { return _heatfrac; };
    __host__ __device__ float& hin() { return _hin; };
    __host__ __device__ float& din() { return _din; };
    __host__ __device__ float& tin() { return _tin; };
};


