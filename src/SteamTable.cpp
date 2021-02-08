//
// Created by JOO IL YOON on 2021/02/01.
//

#include "SteamTable.h"

extern "C" {
    void setTHPressure(const float* press);
    void getTHSatTemperature(float* tm);
    void getTHDensity(const float* h, float* dm);
    void getTHTemperature(const float* h, float* tm);
    void checkTHEnthalpy(const float* h, int* err);
    void getTHEnthalpy(const float* tm, float* h);
}

SteamTable::SteamTable() {}

SteamTable::~SteamTable() {

}

void SteamTable::checkEnthalpy(const float& h, SteamError& err) {
    checkTHEnthalpy(&h, (int*) &err);
}

__host__ __device__ void SteamTable::setPressure(const float& press)
{
    setTHPressure(&press);
}

void SteamTable::getSatTemperature(float& tm) {
    getTHSatTemperature(&tm);
}

void SteamTable::getTemperature(const float& h, float& tm) {
    getTHTemperature(&h, &tm);
}

void SteamTable::getDensity(const float& h, float& dm) {
    getTHDensity(&h, &dm);
}

void SteamTable::getEnthalpy(const float& tm, float& h)
{
    getTHEnthalpy(&tm, &h);
}
