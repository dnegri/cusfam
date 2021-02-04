#pragma once

#include "pch.h"
#include "Geometry.h"
#include "SteamTable.h"

class Feedback {

private:
    Geometry& _g;
    SteamTable& _steam;

    bool _tffrz;
    bool _tmfrz;
    bool _dmfrz;

    float _heatfrac;
    float _hin;

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

    int _ntfpow;
    int _ntfbu;

    float* _tfpow;
    float* _tfbu;
    float* _tftable;

public:
    __host__ __device__ Feedback(Geometry& g, SteamTable& steam);
    __host__ __device__ virtual ~Feedback();

    __host__ __device__ float& ppm0(const int& l) { return _ppm0[l]; };
    __host__ __device__ float& stf0(const int& l) { return _stf0[l]; };
    __host__ __device__ float& tm0(const int& l) { return _tm0[l]; };
    __host__ __device__ float& dm0(const int& l) { return _dm0[l]; };
    __host__ __device__ float& dtf(const int& l) { return _dtf[l]; };
    __host__ __device__ float& dtm(const int& l) { return _dtm[l]; };
    __host__ __device__ float& ddm(const int& l) { return _ddm[l]; };
    __host__ __device__ float& tf(const int& l) { return _tf[l]; };
    __host__ __device__ float& tm(const int& l) { return _tm[l]; };
    __host__ __device__ float& dm(const int& l) { return _dm[l]; };
    __host__ __device__ float& tm(const int& l2d, const int& k) { return _tm[k*_g.nxy()+l2d]; };
    __host__ __device__ float& dm(const int& l2d, const int& k) { return _dm[k*_g.nxy()+l2d]; };

    __host__ __device__ float& chflow(const int& l2d) { return _chflow[l2d]; };
    __host__ __device__ void updateTf(const int& l, const float* pow, const float* bu);

    __host__ __device__ void updateTm(const int& l2d, const float* pow, int& nboiling);

    __host__ __device__ virtual void udpateTf(const float* power, const float* burnup);

    __host__ __device__ virtual void updateTm(const float* power, int& nboiling);

    __host__ __device__ void
    setTfTable(const int& ntfpow, const int& ntfbu, const float* tfpow, const float* tfbu, const float* tftable);
    __host__ __device__ float& tfbu(const int& i) { return _tfbu[i]; };
    __host__ __device__ float& tfpow(const int& i) { return _tfpow[i]; };
    __host__ __device__ float& tftable(const int& ibu, const int& ipow) { return _tfpow[ipow*_ntfbu+ibu]; };


};


