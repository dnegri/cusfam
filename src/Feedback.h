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

    float* _tf;
    float* _tm;
    float* _dm;

    int _ntfpow;
    int _ntfbu;

    float* _tfpow;
    float* _tfbu;
    float* _tftable;

public:
    Feedback(Geometry& g, SteamTable& steam);
    __host__ __device__ float& tf(const int& l) { return _tf[l]; };
    __host__ __device__ float& tm(const int& l) { return _tm[l]; };
    __host__ __device__ float& dm(const int& l) { return _tm[l]; };
    __host__ __device__ float& tm(const int& l2d, const int& k) { return _tm[k*_g.nxy()+l2d]; };
    __host__ __device__ float& dm(const int& l2d, const int& k) { return _dm[k*_g.nxy()+l2d]; };
    __host__ __device__ float& chflow(const int& l2d) { return _chflow[l2d]; };
    __host__ __device__ void updateTf(const int& l, const float* pow, const float* bu);

    __host__ __device__ void updateTm(const int& l2d, const float* pow, int& nboiling);

    __host__ __device__ virtual void udpateTf();

    __host__ __device__ virtual void updateTm();

    __host__ __device__ void
    setTfTable(const int& ntfpow, const int& ntfbu, const float* tfpow, const float* tfbu, const float* tftable);
    __host__ __device__ float& tfbu(const int& i) { return _tfbu[i]; };
    __host__ __device__ float& tfpow(const int& i) { return _tfpow[i]; };
    __host__ __device__ float& tftable(const int& ibu, const int& ipow) { return _tfpow[ipow*_ntfbu+ibu]; };


};


