#pragma once
#include "pch.h"

enum SteamError {
    NO_ERROR,
    STEAM_TABLE_ERROR_MAXENTH
};

class SteamTable : public Managed {

public:
    __host__ __device__ void checkEnthalpy(const float& h, SteamError& err);
    __host__ __device__ void setPressure(const float& press);
    __host__ __device__ void getSatTemperature(float& tm);
    __host__ __device__ void getTemperature(const float& h, float& tm);
    __host__ __device__ void getDensity(const float& h, float& dm);
    __host__ __device__ void getEnthalpy(const float& tm, float& h);
    virtual ~SteamTable();

    SteamTable();
};


