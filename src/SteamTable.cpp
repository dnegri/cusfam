//
// Created by JOO IL YOON on 2021/02/01.
//

#include "SteamTable.h"

SteamTable::SteamTable() {}

SteamTable::~SteamTable() {

}

SteamError SteamTable::checkEnthalpy(const float& h) {
    return STEAM_TABLE_ERROR_MAXENTH;
}

const float& SteamTable::getSatTemperature() {
    return _sattm;
}

float SteamTable::getTemperature(const float& h) {
    return 0;
}

float SteamTable::getDensity(const float& h) {
    return 0;
}
