#pragma once
#include "pch.h"

enum SteamError {
    NO_ERROR,
    STEAM_TABLE_ERROR_MAXENTH
};

class SteamTable : public Managed {

    float _sattm;
public:
    __host__ __device__ SteamError checkEnthalpy(const float& h);
    __host__ __device__ const float& getSatTemperature();
    __host__ __device__ float getTemperature(const float& h);
    __host__ __device__ float getDensity(const float& h);

    virtual ~SteamTable();

    SteamTable();
};


