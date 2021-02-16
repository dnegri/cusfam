#pragma once
#include "pch.h"
#include "SteamTable.h"

class SteamTableCuda : public SteamTable {
private:
	SteamTable* _steam_cpu;
public:
	__host__ SteamTableCuda(SteamTable& steam);
	__host__ virtual ~SteamTableCuda();

	__host__ void setPressure(const float& press) override;
	__host__ SteamTable& getSteamTableCPU() { return *_steam_cpu; };
};


